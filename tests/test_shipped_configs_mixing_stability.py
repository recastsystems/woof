# tests/test_shipped_configs_mixing_stability.py
"""No config under ``configs/`` may ship in the exposed-mixing state.

The 1.6.0 anisotropic-mixing instability was closed with a criterion and
a warning, and then the shipped example configs kept the setting that
trips it: the two 250 m nested-LES trees at
``mix_upper_bound*(dz_max/dx)^2 = 0.702`` and both 100 m tornado trees at
4.23, 17x the limit -- the last of which aborted a run at step 5467 with
w = 239 m/s.  Nothing in the repository would have noticed, because the
criterion was checked against a configuration only when somebody loaded
one and read the stderr.

This file is the thing that notices.  It walks every TOML under
``configs/``, resolves it through the loader a user's run would use, and
requires the criterion to hold on every domain.  The remedy when it
fails is the one the advisory names and the one attempt #3 ruled on:
``mix_isotropic = 1``, or a ``mix_upper_bound`` under the admitted
value.

FROZEN RECORDS ARE THE ONE EXEMPTION, AND THEY ARE PINNED BY CONTENT.
A config a committed receipt was produced under is a record: it has to
keep saying what it said, exposed setting and all.  Each one is listed
in :data:`FROZEN_RECORDS` with the sha256 of its bytes, so the exemption
cannot be widened by editing a record and cannot be reused to smuggle a
new config into the exposed state -- a new file has no entry, and an
edited file no longer matches its entry.

CPU only, no card, no data.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from woof.config import (EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT, RunConfig,
                          UNRESOLVED_ANISOTROPIC_DEPTH_MARK,
                          anisotropic_w_mixing_advice)
from woof.experiment import (anisotropic_w_mixing_exposure,
                              is_experiment_toml, load_experiment)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

#: The configs that are records rather than menu items, each pinned to
#: the bytes its record was produced under.  Keys are POSIX paths
#: relative to the repository root; values are (sha256, why).
#:
#: Adding an entry here is a deliberate act with a receipt behind it.
#: There is no wildcard and no directory-level exemption on purpose:
#: ``configs/frozen/`` is where archived bytes live BY CONVENTION, and
#: this table is what makes it true.
FROZEN_RECORDS: dict[str, tuple[str, str]] = {
    # ALL FIVE WERE RE-PINNED AT 1.8.8, for the only reason a frozen
    # record may move: the file could no longer LOAD at its old bytes.
    # The constant-downward-longwave guard refuses a longwave-free suite
    # under Noah unless the experiment declares it, and every record in
    # this table runs exactly that pairing, so each gained
    # `constant-downward-longwave-v1` -- beside the nocturnal token,
    # where one was already carried -- with the JUSTIFY block the
    # convention requires.  NO PHYSICS SELECTOR MOVED: the runs these
    # bytes are the record of are still reproducible from them, and what
    # was added is a declaration of what those selectors always meant.
    # Leaving the old digests would freeze the files into being
    # unloadable, which is not what freezing them is for.
    #
    # Each entry names its AS-RUN digest as well.  That is the number the
    # committed run receipts carry in `experiment_config`, no receipt was
    # edited, and the bytes behind it stay retrievable from git -- so the
    # record of what ran and the file a reproduction points at are both
    # exact, and they are two different numbers from 1.8.8 on.
    "configs/frozen/les_nest_250m_km3.toml": (
        # RE-PINNED for WOOF 1.0.0: a comment names the engine WOOF-ArWen; a
        # comment cites the unpublished record folder by a neutral name. No
        # value changed; the as-run digest is unchanged.
        "b3b1104876325dea10d412ebeb740738d2afdeb1772847ba9e183c14e878ffc6",
        "the nested-LES 250 m tree the inflow-fetch run_a/run_b and "
        "grayzone p4/r1 receipts ran, plus its 1.8.8 declaration; as-run "
        "digest 2a3a279b6d..., which is what those receipts and "
        "docs/internal/receipts/les/INFLOW-FETCH-D90-2026-08-03.md "
        "name, at `git show 4152fcb31:configs/les_nest_250m_km3.toml`",
    ),
    "configs/frozen/les_nest_250m_grayzone.toml": (
        # RE-PINNED for WOOF 1.0.0: a comment cites the unpublished record
        # folder by a neutral name. No value changed; the as-run digest is
        # unchanged.
        "8db558076a9779d28a0cc3b538dc85a0c128668737b860c0e2349ebeabbbf094",
        "the bytes the P4 gray-zone campaign's R2 and R3 arms ran, plus "
        "their 1.8.8 declaration; as-run digest d3eb6c70c0..., at "
        "`git show 4152fcb31:configs/les_nest_250m_grayzone.toml`",
    ),
    # Re-pinned at 2.9.0 for a comment-only edit: one comment line reworded so
    # the file names no person (public-push hygiene).  tomllib reads the same
    # document from both byte strings; the as-run digest is unchanged.
    "configs/frozen/les_tornado_100m_mayfield_20211210.toml": (
        # RE-PINNED for WOOF 1.0.0: a comment no longer names a person or cites
        # an internal document; a comment names the engine WOOF-ArWen; a comment
        # cites the unpublished record folder by a neutral name. No value
        # changed; the as-run digest is unchanged.
        "445e3f725f90eb84fdd627f013eef08d108254cf58291d2b149a81fbd46155c3",
        "the tornado-LES attempt #1 configuration, plus its 1.8.8 "
        "declaration; as-run digest 7aab68c6df..., which is what "
        "docs/les/ATTEMPT1-EXPECTATIONS.md registered before the run, at "
        "`git show 4152fcb31:configs/"
        "les_tornado_100m_mayfield_20211210.toml`",
    ),
    "configs/les_tornado_100m_mayfield_20211210_attempt2.toml": (
        # RE-PINNED for WOOF 1.0.0: a comment cites the unpublished record
        # folder by a neutral name. No value changed; the as-run digest is
        # unchanged.
        "8b27d7426d85fdf228acdd40bf139dc419f2147b9fd04594bb81af85734a453e",
        "attempt #2 revision a: the run whose arm a is the record of both "
        "the crash and the tornadic vortex that motivated the re-placement",
    ),
    "configs/les_tornado_100m_mayfield_20211210_attempt2b.toml": (
        # RE-PINNED for WOOF 1.0.0: a comment cites the unpublished record
        # folder by a neutral name. No value changed; the as-run digest is
        # unchanged.
        # RE-PINNED 2026-09-20: c065da2d4 (2026-09-09) changed one word of
        # one comment in this record (an owner's name became "the
        # project") and no value; the digest below is that commit's
        # bytes (proof/node-reds-276).
        "f02f47f8deded7f961888f39a9c704ecec26fd03d9764d7d7f8281abb411851f",
        "attempt #2 revision b: the run that tripped at step 5467 with "
        "w = 239.48 m/s and produced the criterion; frozen by name in "
        "configs/les_tornado_100m_mayfield_20211210_attempt3.toml:248-255",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exposed_domains(path: Path) -> list[tuple[int, float]]:
    """``[(grid_id, ratio), ...]`` over the limit, for one config file.

    Routed exactly the way a run routes it: an experiment TOML through
    ``load_experiment``, anything else through the legacy loader.

    An experiment TOML that omits ``eta_levels`` is NOT a hole: the
    exposure helper resolves the ladder the model would build from
    ``nz``/``ztop`` and the criterion runs on that.  Skipping it was the
    hole -- see
    :func:`woof.experiment.anisotropic_w_mixing_exposure` and
    :func:`test_an_experiment_with_no_eta_ladder_is_still_judged` below.

    NOR IS A LADDER THAT WILL NOT RESOLVE AT ALL.  Where the depths come
    back non-finite the ratio is ``None`` -- there is no number -- and
    routing on the ratio alone put that case back in the silence the
    resolution work had just taken it out of.  The screen therefore asks
    :func:`woof.config.anisotropic_w_mixing_advice`, which answers with
    an advisory in that case, and reports the domain with an infinite
    ratio: the same convention the legacy branch below already uses, and
    for the same reason -- the config is on the exposed path and nothing
    can say where it sits relative to the limit.

    The legacy path deserves its own sentence.  A legacy RunConfig is
    not routed through the experiment loader at all here -- it is read
    as raw TOML -- so no ladder is resolved for it and no ratio can be
    computed.  A legacy config that SELECTS the exposed path is
    therefore reported with an infinite ratio: not because it is known
    to be unstable, but because on that route nothing can tell anyone
    whether it is.

    It is read as raw TOML rather than through ``load_config``, because
    several files under ``configs/`` carry tables that loader does not
    own (``[ensemble]``, ``[case_data]``, ``[static]``) and refuses
    outright.  A screen that skipped whatever failed to load would be a
    screen with holes in exactly the places nobody looks.
    """

    experiment = _resolve(path)
    if experiment is not None:
        dz_max, exposed, ladder = anisotropic_w_mixing_exposure(experiment)
        out = []
        for domain in exposed:
            ratio, advice = anisotropic_w_mixing_advice(
                where=f"d{domain.grid_id:02d}",
                km_opt=domain.run.km_opt,
                mix_isotropic=domain.run.mix_isotropic,
                mix_upper_bound=domain.run.mix_upper_bound,
                dx=domain.run.dx, dy=domain.run.dy, dz_max=dz_max,
                ladder=ladder)
            if ratio is None:
                if advice is not None:
                    out.append((domain.grid_id, float("inf")))
                continue
            if ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT:
                out.append((domain.grid_id, float(ratio)))
        return out

    import tomllib

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if _raw_selects_the_exposed_path(raw):
        return [(1, float("inf"))]
    return []


def _resolve(path: Path):
    """The resolved experiment tree, or ``None`` when there is not one.

    ``None`` is never "assume it is fine": the caller falls back to the
    raw selector screen, which reports an exposed selector it cannot put
    a number on.  A config that will not resolve is the LAST place to
    give the benefit of the doubt.
    """

    if not is_experiment_toml(path):
        return None
    payload = path.read_bytes()
    import tomllib

    try:
        if "case_data" in tomllib.loads(payload.decode("utf-8")):
            from woof.case_data import load_experiment_case_bytes

            return load_experiment_case_bytes(
                payload, source=str(path), base_dir=path.parent,
                require_inputs=False)[0]
        return load_experiment(path)
    except Exception:
        return None


def _raw_selects_the_exposed_path(raw: dict) -> bool:
    """km_opt 2 or 3 anywhere, with the mixing length left per-axis.

    ``mix_isotropic`` defaults to 0 on ``RunConfig``, so a table that
    selects the closure and says nothing about the length IS on the
    exposed path -- silence is the dangerous value, not the safe one.
    """

    scopes: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            scopes.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(raw)
    root = {key: value for scope in scopes for key, value in scope.items()
            if not isinstance(value, (dict, list))}
    for scope in scopes:
        km_opt = scope.get("km_opt")
        if km_opt is None or int(km_opt) not in (2, 3):
            continue
        isotropic = scope.get("mix_isotropic", root.get("mix_isotropic", 0))
        if int(isotropic) == 0:
            return True
    return False


def scan(root: Path) -> dict[str, list[tuple[int, float]]]:
    """Every config under ``root`` that is over the limit, by rel path."""

    findings: dict[str, list[tuple[int, float]]] = {}
    for path in sorted(root.rglob("*.toml")):
        over = exposed_domains(path)
        if over:
            findings[path.relative_to(root).as_posix()] = over
    return findings


def test_no_shipped_config_ships_in_the_exposed_mixing_state():
    """The durable half of the 1.6.0 fix.

    A config arriving here in the exposed state is the defect this file
    exists to stop, so the failure names the file, the domain, the
    number and the two remedies rather than asserting a bare False.
    """

    findings = scan(CONFIGS)
    offenders = {
        f"configs/{rel}": over for rel, over in findings.items()
        if f"configs/{rel}" not in FROZEN_RECORDS
    }
    assert not offenders, (
        "these configs run km_opt 2 or 3 with mix_isotropic = 0 on a grid "
        "whose layers are deep enough that the horizontal mixing of w is "
        "handed a coefficient built on the layer depth: "
        f"mix_upper_bound*(dz_max/dx)^2 above "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT} inverts a 2dx mode and "
        "above 0.5 grows one.\n"
        + "\n".join(
            f"  {name}: " + ", ".join(
                f"d{gid:02d} ratio {ratio:.3g}" for gid, ratio in over)
            for name, over in sorted(offenders.items()))
        + "\nSet mix_isotropic = 1 on the named domains (the remedy ruled "
          "on in configs/les_tornado_100m_mayfield_20211210_attempt3.toml"
          ":238) or lower mix_upper_bound.  If this file is a record of a "
          "committed run rather than a configuration to run, add it to "
          "FROZEN_RECORDS with its sha256 and the receipt it belongs to.")


def test_the_shipped_les_trees_are_the_ones_this_was_written_for():
    """Named-file coverage, so the walk cannot pass by finding nothing.

    A glob that quietly stops matching is the classic way a repository
    check goes green forever.  These four are the files the 1.6.0 defect
    was actually in; if one is renamed away, this fails and somebody
    decides deliberately.
    """

    for name in ("les_nest_250m_km3.toml",
                 "les_nest_250m_grayzone.toml",
                 "les_tornado_100m_mayfield_20211210.toml",
                 "les_tornado_100m_dodgecity_20160524.toml"):
        path = CONFIGS / name
        assert path.is_file(), name
        assert exposed_domains(path) == [], name
        # And not by accident of geometry: every LES child is on the
        # isotropic length, which is what takes them off the path.
        experiment = load_experiment(path)
        les = [d for d in experiment.domains if d.run.km_opt in (2, 3)]
        assert les, f"{name} has no km_opt 2/3 domain left to check"
        assert all(d.run.mix_isotropic == 1 for d in les), name


def test_active_moist_configs_keep_cq_and_frozen_records_keep_their_pins():
    """Live low-order suites use CQ; recorded trajectories keep their bytes."""
    import tomllib

    keys = ("moist", "mp_physics", "moist_cq")
    defaults = {key: RunConfig.__dataclass_fields__[key].default
                for key in keys}
    checked: set[str] = set()
    frozen: set[str] = set()
    for path in sorted(CONFIGS.rglob("*.toml")):
        rel = path.relative_to(REPO).as_posix()
        record = FROZEN_RECORDS.get(rel)
        if record is not None:
            assert _sha256(path) == record[0], rel
            experiment = _resolve(path)
            assert experiment is not None, rel
            assert all(domain.run.moist_cq is False
                       for domain in experiment.domains), rel
            frozen.add(rel)
            continue
        if is_experiment_toml(path):
            experiment = _resolve(path)
            if experiment is None:
                # CQ is a physics policy. Resolve it without a case-data
                # path whose environment variable may be absent here.
                from woof.experiment import build_experiment
                physics_document = tomllib.loads(path.read_text(encoding="utf-8"))
                for companion in ("case_data", "fetch", "static"):
                    physics_document.pop(companion, None)
                experiment = build_experiment(physics_document, source=str(path))
            runs = [(domain.grid_id,
                     {key: getattr(domain.run, key) for key in keys})
                    for domain in experiment.domains]
        else:
            # Legacy files put RunConfig values in separate tables. Only
            # these three settings matter to the CQ policy check.
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            selected = {**defaults,
                        **{key: raw[key] for key in keys if key in raw}}
            for table in raw.values():
                if isinstance(table, dict):
                    selected.update({key: table[key]
                                     for key in keys if key in table})
            runs = [(1, selected)]
        for grid_id, run in runs:
            if run["moist"] and run["mp_physics"] in (0, 1, 6):
                checked.add(rel)
                assert run["moist_cq"] is True, (
                    f"{rel} d{grid_id:02d} disables WRF's moisture pressure "
                    "correction; set moist_cq = true in the live config")
    assert frozen == set(FROZEN_RECORDS)
    assert {
        "configs/initdemo_bubble_off.toml",
        "configs/initdemo_bubble_on.toml",
        "configs/les_nest_250m_grayzone.toml",
        "configs/les_nest_250m_km3.toml",
        "configs/les_tornado_100m_dodgecity_20160524.toml",
        "configs/les_tornado_100m_mayfield_20211210.toml",
        "configs/les_tornado_100m_mayfield_20211210_attempt3.toml",
        "configs/les_tornado_100m_mayfield_20211210_attempt3_fine30s.toml",
    } <= checked


def test_every_frozen_record_still_hashes_to_its_pin():
    """The exemption is on bytes, not on a path.

    Without this, ``FROZEN_RECORDS`` would be a list of filenames anyone
    could edit freely -- the opposite of frozen.
    """

    for rel, (digest, why) in sorted(FROZEN_RECORDS.items()):
        path = REPO / rel
        assert path.is_file(), f"{rel} is pinned but missing ({why})"
        assert _sha256(path) == digest, (
            f"{rel} is a frozen record of a committed run and its bytes "
            f"changed: expected sha256 {digest}, found {_sha256(path)}. "
            f"It is exempt from the mixing criterion precisely because it "
            f"must keep saying what it said ({why}); edit the live config "
            f"instead.")


def test_no_frozen_record_is_exempt_for_nothing():
    """Every exemption must actually be exercised by the criterion.

    A pin that no longer names an exposed config is a stale exemption,
    and stale exemptions are how an allowlist grows into a bypass.
    """

    for rel in sorted(FROZEN_RECORDS):
        over = exposed_domains(REPO / rel)
        assert over, (
            f"{rel} no longer violates the criterion, so its "
            f"FROZEN_RECORDS entry is dead weight -- delete the entry")


# --- the controls: the scan has to be able to fail, and not otherwise ---


#: A model top past the analytic base state's ~24.6 km ceiling, so
#: ``woof.core.grid.analytic_base_pressure`` returns ``None`` for it and
#: no ladder can be resolved from ``nz``/``ztop`` at all.  This is the one
#: shape that reaches the criterion's third outcome: on the exposed path,
#: with no number.
ABOVE_THE_ANALYTIC_CEILING = 26000.0


def _write_tree(directory: Path, *, iso: int, ladder: bool = True,
                ztop: float = 20000.0) -> Path:
    """A minimal real-shaped tree at 100 m with a 650 m top layer.

    ``ladder=False`` drops the two keys that declare the eta interfaces,
    which is the shape the scan used to walk straight past.  ``ztop``
    raised to :data:`ABOVE_THE_ANALYTIC_CEILING` is the shape whose
    depths will not resolve even then.
    """

    nz = 8
    eta = [1.0 - i / nz for i in range(nz)] + [0.0]
    rungs = (f"p_top = 50000.0\neta_levels = {eta!r}\n") if ladder else ""
    text = textwrap.dedent("""\
        [experiment]
        name = "control"
        start_time = 1974-04-03T12:00:00
        run_seconds = 600.0
        restart_interval_s = 0.0

        [shared]
        nz = {nz}
        ztop = {ztop}
        {rungs}km_opt = 3
        ra_lw_physics = 0
        ra_sw_physics = 0
        bl_pbl_physics = 0
        mix_upper_bound = 0.1
        mix_isotropic = {iso}

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 40
        ny = 40
        time_step = 2
        dx = 100.0
        history_interval_s = 600.0
        """).format(nz=nz, ztop=repr(float(ztop)), rungs=rungs, iso=iso)
    path = directory / "control.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_scan_catches_an_exposed_config_dropped_into_the_tree(tmp_path):
    """A check that cannot fail is not a check.

    This is the mutation the real defect was: a config authored with the
    per-axis mixing length on a grid whose layers dwarf its spacing.
    """

    _write_tree(tmp_path, iso=0)
    findings = scan(tmp_path)
    assert list(findings) == ["control.toml"]
    (gid, ratio), = findings["control.toml"]
    assert gid == 1
    assert ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT


def test_the_scan_is_silent_on_the_same_grid_with_one_mixing_length(tmp_path):
    """The negative control, on the SAME geometry.

    A guard that fires on the legitimate configuration is worse than the
    defect, so the pair differs by exactly the one key the remedy moves.
    """

    _write_tree(tmp_path, iso=1)
    assert scan(tmp_path) == {}


def test_an_experiment_with_no_eta_ladder_is_still_judged(tmp_path):
    """The blind spot this screen shipped with, closed.

    Same tree, same selectors, same 100 m spacing -- only the two keys
    that WRITE OUT the eta interfaces are gone.  Until 2026-08-09 that
    was enough to walk the whole screen past it: the exposure helper
    returned no domains, ``scan`` found nothing, and this file reported
    the tree as clean.  A config is not safer for having declined to
    spell its ladder out, and the ladder is not unknown -- ``nz`` and
    ``ztop`` are required keys and the model builds the interfaces from
    them.

    The resolved ladder is COARSER than the declared one (8 uniform
    layers to 20 km rather than a 50 kPa lid), so the ratio here is
    larger than the declared-ladder control's above, not smaller: the
    thing the silence was hiding was the worse of the two.
    """

    _write_tree(tmp_path, iso=0, ladder=False)
    findings = scan(tmp_path)
    assert list(findings) == ["control.toml"]
    (gid, ratio), = findings["control.toml"]
    assert gid == 1
    assert ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT

    declared = _write_tree(tmp_path, iso=0, ladder=True)
    (_, declared_ratio), = exposed_domains(declared)
    assert ratio > declared_ratio


def test_the_ladderless_tree_is_silent_with_one_mixing_length(tmp_path):
    """Its negative control, on the same missing ladder.

    Resolving the depths must not turn the screen into "everything
    without an eta table is a finding"; the remedy still silences it.
    """

    _write_tree(tmp_path, iso=1, ladder=False)
    assert scan(tmp_path) == {}


def test_a_tree_whose_depths_will_not_resolve_is_reported_not_skipped(
        tmp_path):
    """The blind spot BEHIND the previous one, closed 2026-08-09.

    Resolving the ladder from ``nz``/``ztop`` closed the case where a
    config declines to write its interfaces.  It left the case where the
    resolution itself fails -- a model top above the analytic base
    state's ceiling -- and that one went quiet at every door, because the
    exposure's ``inf`` sentinel went into ``anisotropic_w_mixing_ratio``
    and came back out as ``None``, which is the same value the function
    returns for "not on the exposed path at all".

    It is not the same thing, and this is the difference: the tree below
    SELECTS the per-axis mixing length at 100 m spacing and no depth
    exists to judge it by.  The screen reports it with an infinite ratio,
    and the advisory door says in words that the number is missing.
    """

    _write_tree(tmp_path, iso=0, ladder=False,
                ztop=ABOVE_THE_ANALYTIC_CEILING)
    findings = scan(tmp_path)
    assert list(findings) == ["control.toml"]
    (gid, ratio), = findings["control.toml"]
    assert gid == 1
    assert ratio == float("inf")

    # And the infinity is the no-number branch rather than a real ratio
    # that happens to overflow: no depth resolved, and the sentence the
    # loader and `woof check` print says exactly that.
    experiment = load_experiment(tmp_path / "control.toml")
    dz_max, exposed, ladder = anisotropic_w_mixing_exposure(experiment)
    assert dz_max == float("inf")
    (domain,) = exposed
    number, advice = anisotropic_w_mixing_advice(
        where=f"d{domain.grid_id:02d}", km_opt=domain.run.km_opt,
        mix_isotropic=domain.run.mix_isotropic,
        mix_upper_bound=domain.run.mix_upper_bound,
        dx=domain.run.dx, dy=domain.run.dy, dz_max=dz_max, ladder=ladder)
    assert number is None
    assert UNRESOLVED_ANISOTROPIC_DEPTH_MARK in advice
    assert "representable ceiling" in advice
    assert "ADVISORY, not a refusal" in advice


def test_the_unresolvable_tree_is_silent_with_one_mixing_length(tmp_path):
    """Its negative control, on the same unresolvable model top.

    The finding is the per-axis length with no depth to bound it, not the
    model top by itself -- a tree that cannot resolve its ladder but
    builds ONE mixing length is off the path entirely and there is
    nothing to say about it.
    """

    _write_tree(tmp_path, iso=1, ladder=False,
                ztop=ABOVE_THE_ANALYTIC_CEILING)
    assert scan(tmp_path) == {}


def test_the_unresolvable_tree_is_silent_on_a_horizontal_only_selector(
        tmp_path):
    """The other off-path control at the same model top."""

    path = _write_tree(tmp_path, iso=0, ladder=False,
                       ztop=ABOVE_THE_ANALYTIC_CEILING)
    path.write_text(path.read_text(encoding="utf-8")
                    .replace("km_opt = 3", "km_opt = 4"), encoding="utf-8")
    assert scan(tmp_path) == {}


def test_both_routes_report_the_grid_whose_depths_will_not_resolve(tmp_path):
    """The two routes must not disagree about the same physical grid.

    The legacy raw-TOML branch has always reported an exposed selector it
    could put no number on, with an infinite ratio.  The experiment route
    reached the identical situation -- exposed selector, no resolvable
    depth -- and reported nothing, so the better-informed route was the
    quieter one.  Same selectors, same spacing, same model top, one
    answer.
    """

    experiment_dir = tmp_path / "experiment"
    legacy_dir = tmp_path / "legacy"
    experiment_dir.mkdir()
    legacy_dir.mkdir()

    tree = _write_tree(experiment_dir, iso=0, ladder=False,
                       ztop=ABOVE_THE_ANALYTIC_CEILING)
    legacy = legacy_dir / "legacy.toml"
    legacy.write_text(textwrap.dedent(f"""\
        [grid]
        nx = 40
        ny = 40
        nz = 8
        dx = 100.0
        dy = 100.0
        ztop = {ABOVE_THE_ANALYTIC_CEILING!r}

        [dynamics]
        km_opt = 3
        mix_isotropic = 0

        [run]
        dt = 1.0
        run_seconds = 10.0
        """), encoding="utf-8")

    assert exposed_domains(legacy) == [(1, float("inf"))]
    assert exposed_domains(tree) == exposed_domains(legacy)


def test_the_scan_is_silent_on_a_horizontal_only_selector(tmp_path):
    """km_opt = 4 builds one coefficient on the horizontal spacing.

    Every non-LES config in the tree is on this selector, so a false
    positive here would have failed the whole repository.
    """

    path = _write_tree(tmp_path, iso=0)
    path.write_text(path.read_text(encoding="utf-8")
                    .replace("km_opt = 3", "km_opt = 4"), encoding="utf-8")
    assert scan(tmp_path) == {}


def _write_legacy(tmp_path, km_opt):
    """One valid legacy fixture for exposed and horizontal-only selectors."""
    text = (
        '[grid]\n'
        'nx = 40\n'
        'ny = 40\n'
        'nz = 40\n'
        'dx = 100.0\n'
        'dy = 100.0\n'
        'ztop = 20000.0\n'
        '\n'
        '[dynamics]\n'
        'km_opt = 3\n'
        'mix_isotropic = 0\n'
        '\n'
        '[run]\n'
        'dt = 1.0\n'
        'run_seconds = 10.0\n'
    )
    (tmp_path / "legacy.toml").write_text(
        text.replace("km_opt = 3", f"km_opt = {km_opt}"), encoding="utf-8")


def test_the_scan_reports_a_legacy_config_that_selects_the_exposed_path(
        tmp_path):
    """The legacy route cannot compute the ratio, so it reports the
    selector instead -- and says so with an infinite one."""

    _write_legacy(tmp_path, 3)
    findings = scan(tmp_path)
    assert list(findings) == ["legacy.toml"]
    assert findings["legacy.toml"][0][1] == float("inf")


def test_the_legacy_route_is_silent_on_a_horizontal_only_selector(tmp_path):
    """Its negative control: every legacy config in the tree runs
    km_opt = 4 and none of them may be reported."""

    _write_legacy(tmp_path, 4)
    assert scan(tmp_path) == {}


@pytest.mark.parametrize("rel", sorted(FROZEN_RECORDS))
def test_a_frozen_record_would_be_caught_if_it_were_not_pinned(rel):
    """The exemption is doing work, and the scan is what it exempts.

    Each pinned file is over the limit, which is the only reason it
    needs an entry; this states that plainly per file so a reader can
    see the allowlist is a list of real findings, not of convenience.
    """

    over = exposed_domains(REPO / rel)
    assert over
    assert max(ratio for _, ratio in over) > \
        EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
