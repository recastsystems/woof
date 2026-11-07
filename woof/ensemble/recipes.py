"""Source-neutral ensemble rosters and acquisition requests.

These are input recipes, not claims of calibrated forecast uncertainty. Every
member carries one source trajectory through initial and boundary valid times.
Numerical decoding, mapping and initialization remain in the native preparer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from woof.ensemble.seeds import member_seed

CONTRACT = "gpuwm-ensemble-source-recipe.v1"
RECIPE_KINDS = ("wofs", "input-ensemble", "recentered", "time-lagged", "multi-model", "surface-state", "member-roster", "control")


def _utc(value: datetime, *, whole_hour: bool = True) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("source times need an explicit UTC offset to align boundary valid times")
    value = value.astimezone(timezone.utc)
    if whole_hour and (value.minute or value.second or value.microsecond):
        raise ValueError("source recipes need whole-hour valid times")
    return value


@dataclass(frozen=True)
class SourceTrajectory:
    source: str
    cycle: datetime
    member: str | None = None

    def __post_init__(self):
        from woof.source_adapters import get_source_adapter, canonical_source_token, source_for_cycle
        from woof.forcing_member import member_contract
        object.__setattr__(self, "source", source_for_cycle(self.source, self.cycle))
        adapter = get_source_adapter(self.source)
        if not adapter.runnable:
            raise ValueError(f"{adapter.source_id} has no complete initialization route: {adapter.composition_requirement}")
        object.__setattr__(self, "source", canonical_source_token(self.source))
        object.__setattr__(self, "cycle", _utc(self.cycle))
        contract = member_contract(adapter.source_id, self.member)
        if contract is not None:
            object.__setattr__(self, "member", contract[2])
        elif self.member is not None and adapter.selection_owner:
            # The row names the module that owns this source's member
            # selection, the seam the preparation chain verifies through.
            from importlib import import_module
            selected = import_module(adapter.selection_owner).validate_member(self.member, cycle=self.cycle)
            object.__setattr__(self, "member", str(selected))
        elif adapter.acquisition_owner:
            from importlib import import_module
            validate = getattr(import_module(adapter.acquisition_owner), "validate_member", None)
            if validate is not None:
                selected = validate(adapter.source_id, self.member)
                if len(selected) != 1:
                    raise ValueError("one source trajectory needs exactly one source member")
                object.__setattr__(self, "member", str(selected[0]))
            elif self.member is not None:
                raise ValueError(f"{adapter.source_id} has no verified member grammar")
        elif self.member is not None:
            raise ValueError(f"{adapter.source_id} has no verified member grammar")

    @property
    def identity(self) -> str:
        doc = (self.source, self.cycle.isoformat(), self.member)
        return hashlib.sha256(json.dumps(doc, separators=(",", ":")).encode()).hexdigest()

    @property
    def model(self) -> str:
        """The model that produced this trajectory, from its source row.

        Several source ids can be file products of one model run (its
        native levels, its pressure levels, a mapped form of either).  The
        registry row names that model as ``upstream_model_id``; a row that
        names none is its own model.
        """
        from woof.source_adapters import get_source_adapter
        adapter = get_source_adapter(self.source)
        return adapter.model_key or adapter.upstream_model_id or adapter.source_id

    @property
    def model_run(self) -> tuple[str, str, str | None]:
        """What makes two trajectories the same forecast: model, cycle, member."""
        return (self.model, self.cycle.isoformat(), self.member)

    def window(self, start: datetime, end: datetime, *,
               native_bracketing: bool = False) -> tuple[int, int]:
        """Published lead bounds, optionally enclosing native interpolation.

        Bracketing changes acquisition bounds only. It never changes the
        requested forecast window, the source cycle or its population.
        Analysis trajectories retain their distinct valid-time contract.
        """
        from woof.source_cycles import cycle_grid_for
        from woof.source_adapters import get_source_adapter
        import math
        start, end = _utc(start), _utc(end, whole_hour=not native_bracketing)
        if end <= start or start < self.cycle:
            raise ValueError("a source trajectory must cover the complete initial and boundary interval")
        first = int((start - self.cycle).total_seconds()) // 3600
        # Acquisition must enclose a fractional forecast end, never floor it
        # to the initial knot. Only native interpolation admits that end.
        last = math.ceil((end - self.cycle).total_seconds() / 3600)
        adapter = get_source_adapter(self.source)
        if adapter.time_axis == "supplied_times":
            # This describes reuse of caller-supplied fields. Their native
            # manifest must validate actual coverage before preparation; no
            # source cycle, cadence or public acquisition is invented here.
            return first, last
        grid = cycle_grid_for(self.source)
        if adapter.time_axis == "analysis_times":
            if self.cycle != start:
                raise ValueError(
                    f"{self.source} publishes successive analyses at their valid times; "
                    "the trajectory must begin at the requested start, because an earlier "
                    "analysis label cannot create a distinct time-lagged forecast")
            cadence = adapter.forcing_interval_seconds
            if (cadence is None or not math.isfinite(cadence) or cadence <= 0
                    or cadence % 3600):
                raise ValueError(
                    f"{self.source} needs a declared whole-hour analysis cadence "
                    "to identify its initial and boundary valid times")
            elapsed = math.ceil((end - start).total_seconds())
            if elapsed % cadence and not native_bracketing:
                raise ValueError(f"{self.source} analysis window does not end on its declared "
                                 f"{int(cadence)} s cadence; native temporal interpolation is required")
            if elapsed % cadence:
                # A window shorter than the analysis spacing (a 1-3 h run on
                # 6-hourly ERA5) is bracketed by the next analysis, exactly
                # as the single-run door fetches it: the boundary tendency
                # interpolates between the two knots.  This refused before
                # with "native temporal interpolation is required", which the
                # ordinary ERA5 run already does.
                elapsed += int(cadence) - elapsed % int(cadence)
                last = first + elapsed // 3600
            if grid is not None:
                knots = (start + timedelta(seconds=offset)
                         for offset in range(0, elapsed + 1, int(cadence)))
                if any(knot.hour not in grid.analysis_valid_hours for knot in knots):
                    raise ValueError(
                        f"{self.source} does not declare the requested analysis valid-time knots")
            return first, last
        if grid is None or self.cycle.hour not in grid.hours:
            raise ValueError(f"{self.source} does not declare the selected initialization hour")
        horizon = grid.horizon(self.cycle)
        if horizon is None:
            horizon = adapter.max_forecast_hour
        if last > horizon:
            raise ValueError(f"{self.source} cycle {self.cycle.isoformat()} ends at f{horizon}; requested boundary needs f{last}")
        from woof.fetch_routes import ladder_for, table_route
        route = table_route(self.source)
        if route is not None:
            leads = tuple(lead for lead in ladder_for(route, self.cycle) if lead <= horizon)
            aligned = first in leads and last in leads
        else:
            cadence = adapter.forcing_interval_seconds
            aligned = cadence is not None and all((lead * 3600) % cadence == 0 for lead in (first, last))
            leads = (() if cadence is None or cadence <= 0 or cadence % 3600 else
                     tuple(range(0, horizon + 1, int(cadence) // 3600)))
            if adapter.forecast_time_owner is not None and cadence is not None and not cadence % 3600:
                # Some legacy native routes publish a finer ladder than
                # their default spacing. Their own ordinary fetch owner
                # permits, for example, GFS f001/f004/.../f013; modulo the
                # default three-hour cadence would falsely refuse it.
                from woof import fetch
                owner = getattr(fetch, adapter.forecast_time_owner)
                try:
                    native = owner(last - first, self.cycle if adapter.forecast_time_cycle_argument
                                   else int(cadence) // 3600, first)
                    aligned = bool(native) and native[0] == first and native[-1] == last
                except ValueError:
                    aligned = False
        if not aligned and native_bracketing:
            before = tuple(lead for lead in leads if lead <= first)
            after = tuple(lead for lead in leads if lead >= last)
            if before and after:
                return max(before), min(after)
            raise ValueError(
                f"{self.source} cannot bracket f{first}/f{last} with published leads; "
                "native temporal interpolation may not extrapolate")
        if not aligned:
            raise ValueError(
                f"{self.source} does not publish the requested f{first}/f{last} valid-time knots; "
                "native temporal interpolation is required before this trajectory can initialize the common window")
        return first, last

    def fetch_argv(self, start: datetime, end: datetime, root: Path, *,
                   p_top_pa: float | None = None,
                   native_bracketing: bool = False) -> tuple[str, ...]:
        """Arguments for the ordinary fetch door, including its posted loop."""
        from woof.source_adapters import get_source_adapter, fetch_model_top_pa
        from woof.fetch_routes import all_fetchable_sources, acquisition_refusal_reason
        first, last = self.window(start, end, native_bracketing=native_bracketing)
        adapter = get_source_adapter(self.source)
        if adapter.time_axis == "supplied_times" or self.source not in all_fetchable_sources():
            raise ValueError(
                f"{self.source} has no public recipe acquisition: "
                f"{acquisition_refusal_reason(self.source)}; "
                "retain the supplied input manifest for native preparation")
        argv = ["--source", self.source, "--cycle", self.cycle.strftime("%Y-%m-%dT%H"),
                "--hours", str(last - first),
                "--out", str(Path(root) / self.identity), "--as-posted"]
        if adapter.time_axis == "analysis_times":
            argv.extend(("--cadence", str(int(adapter.forcing_interval_seconds) // 3600)))
        else:
            argv.extend(("--forecast-start-hour", str(first)))
            # A roster has source trajectories, not a geographic subset
            # request. Use the common whole-object mode explicitly so a
            # transport cannot inherit a subsetter that needs absent bounds.
            argv.extend(("--mode", "full-file"))
        if adapter.fetch_requires_retrieve:
            argv.append("--retrieve")
        if self.member is not None:
            argv.extend(("--member", self.member))
        top = fetch_model_top_pa(adapter.source_id, p_top_pa)
        if top is not None:
            argv.extend(("--p-top-pa", str(top)))
        return tuple(argv)


@dataclass(frozen=True)
class RecipeMember:
    index: int
    seed: int
    trajectory: SourceTrajectory


@dataclass(frozen=True)
class SourceRecipe:
    kind: str
    base: SourceTrajectory
    start: datetime
    end: datetime
    members: tuple[RecipeMember, ...]
    donor_population: tuple[SourceTrajectory, ...] = ()
    calibration: str = "not calibrated"
    perturbation: dict | None = None
    member_variants: tuple = ()
    selection_roster: tuple = ()

    def describe(self) -> dict:
        document = asdict(self)
        if self.perturbation is None:
            document.pop("perturbation")
        if not self.member_variants:
            document.pop("member_variants")
        document["contract"] = CONTRACT
        # Native preparation consumes one shared geometry owner and distinct
        # atmospheric/boundary products. This is a requirement, not a receipt.
        document["preparation_requirements"] = "shared geometry; byte-verified base; member atmosphere and boundary tables"
        return json.loads(json.dumps(document, default=lambda value: value.isoformat()))

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def acquisitions(self) -> tuple[SourceTrajectory, ...]:
        """Deduplicate identical complete trajectories without losing order."""
        result = {}
        inputs = tuple(member.trajectory for member in self.members)
        if self.kind == "recentered":
            inputs = (self.base,) + self.donor_population + inputs
        for source in inputs:
            result.setdefault(source.identity, source)
        return tuple(result.values())

    def acquisition_window(self, trajectory: SourceTrajectory) -> tuple[int, int]:
        """Keep donor interpolation coverage separate from the base window."""
        if trajectory.identity not in {item.identity for item in self.acquisitions()}:
            raise ValueError("source is outside the frozen recipe acquisition population")
        # Every recipe kind acquires the published knots that enclose its
        # window, as the single-run door does: native preparation
        # interpolates the boundary tendency between them. Before, only wofs
        # and recentered donors bracketed, so a 2 h GFS member-roster
        # (sensitivity) run was refused with "gfs does not publish the
        # requested f0/f2 valid-time knots" (WOOF run_0cd03a9ca0a8f9fb).
        return trajectory.window(self.start, self.end, native_bracketing=True)

    def fetch_argv(self, trajectory: SourceTrajectory, root: Path, *,
                   p_top_pa: float | None = None) -> tuple[str, ...]:
        self.acquisition_window(trajectory)
        return trajectory.fetch_argv(self.start, self.end, root, p_top_pa=p_top_pa,
                                     native_bracketing=True)

    def select_members(self, indices) -> SourceRecipe:
        """Replay or repartition existing members without renumbering seeds."""
        indices = tuple(indices)
        known = {member.index: member for member in self.members}
        if (not indices or len(set(indices)) != len(indices)
                or any(type(index) is not int or index not in known for index in indices)):
            raise ValueError("member selection must name distinct indices in the original roster")
        return replace(self, members=tuple(known[index] for index in indices))


def ensemble_population(source: str, cycle: datetime, *, perturbed_only=False) -> tuple[SourceTrajectory, ...]:
    from woof.source_adapters import get_source_adapter, source_for_cycle
    source = source_for_cycle(source, cycle)
    from woof.source_authorities import packaged_member_grammar
    from woof.member_grammar import load_member_grammar
    adapter = get_source_adapter(source)
    if not adapter.member_set:
        raise ValueError(f"{adapter.source_id} has no verified trajectory-member set")
    grammar = load_member_grammar(packaged_member_grammar(adapter.member_set))
    # Member class is metadata. Neither perturbation ordinal zero nor a
    # producer-specific filename implies a control trajectory.
    members = tuple(sorted(
        (member for member in grammar.members() if not perturbed_only or member.class_name != "control"),
        key=lambda member: (member.class_name != "control", member.member_id)))
    result = tuple(SourceTrajectory(adapter.source_id, cycle, member.member_id) for member in members)
    control_source = getattr(adapter, "ensemble_control_source", None)
    if control_source and not perturbed_only:
        result = (SourceTrajectory(control_source, cycle),) + result
    return result


def build_recipe(*, source: str, cycle: datetime, start: datetime, end: datetime,
                 count: int, base_seed: int, kind: str = "auto",
                 donor: SourceTrajectory | None = None,
                 trajectories: tuple[SourceTrajectory, ...] = (),
                 max_lag_hours: int = 24,
                 member: str | None = None,
                 perturbation: dict | None = None,
                 member_variants: tuple = ()) -> SourceRecipe:
    """Resolve a recipe without downloading data or initializing CUDA.

    Automatic singleton is the unchanged base. An explicit recipe can replay
    one selected source member. Unsupported automatic defaults require a
    recipe choice until observation calibration supplies a measured default.

    ``member`` is the base trajectory's own source member (a config's
    ``[fetch].member``).  A time-lagged roster keeps it at every lagged
    cycle; without it the base takes the source's default member.
    """
    from woof.source_adapters import get_source_adapter
    from woof.source_cycles import cycle_grid_for
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("member count must be a positive integer")
    member_seed(base_seed, 0)
    base = SourceTrajectory(source, cycle, member)
    start, end = _utc(start), _utc(end, whole_hour=kind != "wofs")
    base.window(start, end, native_bracketing=True)
    adapter = get_source_adapter(base.source)
    if kind == "auto":
        if count == 1:
            kind = "control"
        elif adapter.member_set or getattr(adapter, "ensemble_source", None):
            ensemble_id = adapter.source_id if adapter.member_set else adapter.ensemble_source
            ensemble_adapter = get_source_adapter(ensemble_id)
            if not ensemble_adapter.runnable:
                raise ValueError(
                    f"{base.source} publishes ensemble {ensemble_id}, but it lacks a complete "
                    f"initialization route: {ensemble_adapter.composition_requirement}; "
                    "choose a complete recentered donor explicitly while its default remains uncalibrated")
            kind = "input-ensemble"
        else:
            raise ValueError(f"{base.source} has no observation-calibrated automatic recipe; choose recentered, time-lagged or multi-model explicitly")
    if kind not in RECIPE_KINDS:
        raise ValueError(f"unknown source recipe {kind!r}; expected {RECIPE_KINDS}")
    population = ()
    surface = None
    variants = ()
    if perturbation is not None:
        from woof.ensemble.surface_controls import validate_surface_recipe
        surface = validate_surface_recipe(perturbation)
    if kind in ("member-roster", "wofs"):
        from woof.ensemble.member_variants import normalize_member_variants
        if perturbation is not None:
            raise ValueError("member-roster uses each member's own surface controls; omit shared perturbation")
        variants = normalize_member_variants(member_variants, count, ladder=kind == "wofs" or bool(trajectories))
        if trajectories and len(trajectories) != count:
            raise ValueError("member-roster needs one trajectory for each named arm; "
                             "extra or missing trajectories would bind the wrong source to an arm")
        selected = tuple(trajectories) if trajectories else (base,) * count
    elif kind == "surface-state":
        from woof.ensemble.surface_controls import shared_surface_options
        surface = shared_surface_options(perturbation, count)
        selected = (base,) * count
    elif kind == "control":
        selected = (base,) * count
    elif kind == "input-ensemble":
        ensemble = adapter.source_id if adapter.member_set else getattr(adapter, "ensemble_source", None)
        if not ensemble:
            raise ValueError(f"{base.source} has no declared operational ensemble source")
        selected = ensemble_population(ensemble, cycle)
    elif kind == "recentered":
        if donor is None:
            raise ValueError("recentered inputs need a declared ensemble donor cycle to keep initial and boundary anomalies aligned")
        population = ensemble_population(donor.source, donor.cycle, perturbed_only=True)
        selected = population
    elif kind == "time-lagged":
        if isinstance(max_lag_hours, bool) or not isinstance(max_lag_hours, int) or max_lag_hours < 0:
            raise ValueError("max_lag_hours must be a non-negative integer")
        grid = cycle_grid_for(base.source)
        candidates = []
        for lag in range(max_lag_hours + 1):
            candidate_cycle = base.cycle - timedelta(hours=lag)
            if grid is None or candidate_cycle.hour not in grid.hours:
                continue
            candidate = SourceTrajectory(base.source, candidate_cycle, base.member)
            try:
                candidate.window(start, end, native_bracketing=True)
            except ValueError:
                continue
            candidates.append(candidate)
        selected = tuple(candidates)
    else:
        if not trajectories:
            raise ValueError("multi-model inputs need an explicit list of source/cycle trajectories")
        selected = tuple(trajectories)
    if len(selected) < count:
        raise ValueError(f"{kind} supplies {len(selected)} distinct trajectories covering this window, but {count} were requested; repeating trajectories would fabricate ensemble size")
    selected = selected[:count]
    if kind == "multi-model":
        # Breakage it prevents: one model run listed under several of its
        # source ids (its native, pressure-level and mapped file products)
        # would be prepared once per id and published as that many members,
        # with spread that comes only from the preparation.
        # The same source id listed twice is the roster repeat refused below.
        runs: dict = {}
        for item in selected:
            ids = runs.setdefault(item.model_run, [])
            if item.source not in ids:
                ids.append(item.source)
        repeated = {run: ids for run, ids in runs.items() if len(ids) > 1}
        if repeated:
            raise ValueError("; ".join(
                f"{' and '.join(ids)} are file products of one {run[0]} run "
                f"({run[1][:13]}Z" + ("" if run[2] is None else f", member {run[2]}") + ")"
                for run, ids in repeated.items())
                + ": listing one model run under several source ids would inflate "
                  "effective ensemble size. Keep one source id for each model run")
        if len({item.model for item in selected}) < 2:
            raise ValueError("the selected multi-model roster needs at least two distinct source models")
    if kind not in ("control", "surface-state", "member-roster", "wofs") and len({item.identity for item in selected}) != count:
        raise ValueError("the member roster repeats a source trajectory and would inflate effective ensemble size")
    for item in (*selected, *population):
        item.window(start, end, native_bracketing=True)
    return SourceRecipe(kind, base, start, end,
                        tuple(RecipeMember(index, member_seed(base_seed, index), item)
                              for index, item in enumerate(selected)), population,
                         perturbation=surface, member_variants=variants)


def trajectory_time(value, *, refusal=ValueError) -> datetime:
    """A listed trajectory's cycle as a UTC time: a datetime, or ``[fetch]``'s spelling."""
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00").replace("_", "T")
        if len(text) == 13:               # YYYY-MM-DDTHH, the [fetch] spelling
            text += ":00:00"
        value = datetime.fromisoformat(text)
    if not isinstance(value, datetime):
        raise refusal(f"{value!r} is not a UTC time; write a cycle as YYYY-MM-DDTHH")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def load_trajectories(path, *, refusal=ValueError) -> tuple:
    """``--trajectories FILE``: a JSON or TOML list of ``{source, cycle[, member]}``.

    The reader of this planning door and of the forecast doors
    (:mod:`woof.ensemble.recipe_door`, which passes its own ``refusal``
    class).  It lives here because this module is staged into the
    preparation-only wheel and the forecast door is not: read from the
    door, ``python -m woof.ensemble.recipes --trajectories FILE`` was an
    import of a module that wheel does not carry, and the wheel's staging
    refused it as an unresolved internal import.
    """
    import tomllib

    path = Path(path)
    if not path.is_file():
        raise refusal(f"--trajectories {path} does not exist")
    text = path.read_text(encoding="utf-8-sig")
    try:
        document = tomllib.loads(text) if path.suffix.lower() == ".toml" else json.loads(text)
    except (ValueError, tomllib.TOMLDecodeError) as error:
        raise refusal(f"--trajectories {path} is not readable JSON or TOML: {error}") from error
    if isinstance(document, dict):
        document = document.get("trajectories")
    if not isinstance(document, list) or not document:
        raise refusal(
            f"--trajectories {path} needs a list of {{source, cycle}} entries "
            "(a JSON list, or [[trajectories]] tables in TOML)")
    return tuple(document)


def main(argv=None) -> int:
    """Small planning door; emits requests, never implies a forecast ran."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cycle", required=True, type=datetime.fromisoformat)
    parser.add_argument("--hours", required=True, type=int)
    parser.add_argument("--members", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--member", default=None,
                        help="the base trajectory's own source member; a time-lagged "
                             "plan keeps it at every lagged cycle (default: the "
                             "source's default member)")
    parser.add_argument("--recipe", choices=("auto",) + RECIPE_KINDS, default="auto")
    parser.add_argument("--donor")
    parser.add_argument("--donor-cycle", type=datetime.fromisoformat)
    parser.add_argument("--trajectories", metavar="FILE",
                        help="the multi-model member list: a JSON or TOML file of "
                             "{source, cycle[, member]} entries")
    parser.add_argument("--surface-options", metavar="FILE",
                        help="JSON surface-state descriptor with soil_moisture_scale "
                             "and/or sst_offset_k (scalars or [minimum, maximum])")
    args = parser.parse_args(argv)
    donor = None if args.donor is None else SourceTrajectory(args.donor, args.donor_cycle or args.cycle)
    trajectories = ()
    perturbation = (None if args.surface_options is None else
                    json.loads(Path(args.surface_options).read_text(encoding="utf-8")))
    if args.trajectories is not None:
        trajectories = tuple(SourceTrajectory(str(item["source"]), trajectory_time(item["cycle"]),
                                              item.get("member"))
                             for item in load_trajectories(args.trajectories))
    plan = build_recipe(source=args.source, cycle=args.cycle, start=args.cycle,
                        end=args.cycle + timedelta(hours=args.hours), count=args.members,
                        base_seed=args.seed, kind=args.recipe, donor=donor,
                        trajectories=trajectories, member=args.member, perturbation=perturbation)
    print(json.dumps(plan.describe(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
