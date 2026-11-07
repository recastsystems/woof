"""The WoFS-shape any-init ensemble: a member provenance ladder.

``woof ensemble --recipe wofs`` (and a plain ``--members N`` on a source
with no members of its own) lands here.  The ladder fills N members from
the best inputs a source has, in order, and writes down where every member
came from:

1. **source members** -- the source's own ensemble (GEFS: 31; ERA5 EDA when CDS is available), or the source's one trajectory when
   it has none;
2. **time-lagged** -- earlier cycles of the same source (and their members)
   whose forecasts cover the window, nearest cycle first;
3. **perturbation fill** -- the real trajectories again, each branched at
   initialization with a seeded uniform U/V offset, a seeded surface-state
   draw (soil moisture scale, SST offset) and a physics arm from
   :data:`PHYSICS_ARMS`, until N is reached.

None of this is calibrated spread, and the receipt says so member by
member: the provenance label replaces the old "uncalibrated" refusal,
which named no breakage (gate law).  Every knob stays selectable
(:class:`WofsOptions`), and the existing recipes are unchanged.

Forecast sources use the existing per-source preparation chains. ERA5 uses
its config-driven preparation, with independently acquired EDA trajectories
when selected. All routes retain per-member physics and surface controls.

Pure planning: no download, no CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
import hashlib
import json

CONTRACT = "gpuwm-ensemble-wofs-ladder.v1"
DEFAULT_MEMBERS = 36
#: The lane harness's illustrative amplitude (+/-0.5 m/s per member and
#: wind axis), the same default the uniform-wind initializer ships.
DEFAULT_WIND_AMPLITUDE = 0.5
DEFAULT_MAX_LAG_HOURS = 24
DEFAULT_PHYSICS = "fill"
DEFAULT_SEED = 0
#: ``woof ensemble --source ... --init ... --center ...``: the created domain's
#: forecast length and width.  The go parser's defaults are these constants.
DEFAULT_HOURS = 3
DEFAULT_EXTENT_KM = 900.0
#: Seeded surface-state draw ranges for fill members.
SOIL_MOISTURE_RANGE = (0.9, 1.1)
SST_OFFSET_RANGE_K = (-0.5, 0.5)
PHYSICS_MODES = ("fill", "all", "off")

#: The documented physics-diversity set.  Arm 0 is the config's own
#: physics.  The others are convection-permitting microphysics/PBL pairs in
#: the spirit of WoFS's PBL diversity, each with the surface layer its PBL
#: scheme is written against (YSU with the revised MM5 layer 91, MYNN with
#: its own layer 5, MYJ with the Eta layer 2).  Every arm is a set of
#: ordinary single-run selectors, checked by the config loader per member.
#: No arm is aerosol-aware Thompson (mp_physics 28), so every arm also resets
#: the selectors the loader allows only with mp_physics 28: the PBL scalar
#: mixing (config.validate_scalar_pblmix_consumer), the WRF 3.9 NOAA fork
#: generation and its snow fall speed. An HRRR-shaped base config carries all
#: three, so every fill member was refused at preparation ("member 3 cannot be
#: prepared: scalar_pblmix=1 requires ... mp_physics=28", then the fork
#: generation and its snow fall; WOOF 1.1.0 dry cut, 2026-10-19). Each value is
#: the loader's default, so a base config without them is unchanged.
# aer_init_opt and wif_input_opt are mp_physics=28-only too: with them left at the HRRR base value 1, the HRRR
# route refused every fill member (emitted namelist 0 vs config 1; WOOF 1.1.0 dry run, 2026-10-19).
_NO_SCALAR_MIX = {"scalar_pblmix": 0, "thompson_version": "wrf_461", "thompson_fork_snow_fall": "blend",
                  "aer_init_opt": 0, "wif_input_opt": 0}
PHYSICS_ARMS = (
    ("base", {}),
    ("thompson-ysu", {"mp_physics": 8, "bl_pbl_physics": 1, "sf_sfclay_physics": 91, **_NO_SCALAR_MIX}),
    ("thompson-mynn", {"mp_physics": 8, "bl_pbl_physics": 5, "sf_sfclay_physics": 5, **_NO_SCALAR_MIX}),
    ("thompson-myj", {"mp_physics": 8, "bl_pbl_physics": 2, "sf_sfclay_physics": 2, **_NO_SCALAR_MIX}),
    ("nssl2m-ysu", {"mp_physics": 18, "bl_pbl_physics": 1, "sf_sfclay_physics": 91, **_NO_SCALAR_MIX}),
    ("nssl2m-mynn", {"mp_physics": 18, "bl_pbl_physics": 5, "sf_sfclay_physics": 5, **_NO_SCALAR_MIX}),
    ("morrison-myj", {"mp_physics": 10, "bl_pbl_physics": 2, "sf_sfclay_physics": 2, **_NO_SCALAR_MIX}),
    ("wsm6-ysu", {"mp_physics": 6, "bl_pbl_physics": 1, "sf_sfclay_physics": 91, **_NO_SCALAR_MIX}),
)

RUNG_SOURCE = "source-member"
RUNG_LAGGED = "time-lagged"
RUNG_FILL = "perturbation"


class LadderRefusal(ValueError):
    """A WoFS ladder request that cannot make N distinct members."""


def parse_source_spec(value, members=None):
    """CLI comma/count spelling or [ensemble].sources tables, validated together."""
    from woof.source_adapters import ensemble_source_argument
    if isinstance(value, str):
        entries = []
        for part in value.split(","):
            token, sep, count = part.strip().partition(":")
            try:
                entries.append({"source": token, **({"members": int(count)} if sep else {})})
            except ValueError:
                raise LadderRefusal(f"source {token!r} needs a positive whole member count") from None
    elif isinstance(value, (tuple, list)):
        entries = list(value)
    else:
        raise LadderRefusal("sources must be a source spec or a list of {source, members} tables")
    if not entries:
        raise LadderRefusal("sources must name at least one family")
    result = []
    for row in entries:
        if not isinstance(row, dict) or set(row) - {"source", "members"} or "source" not in row:
            raise LadderRefusal("sources needs {source, members} tables")
        try:
            token = ensemble_source_argument(row["source"])
        except (ValueError, AttributeError) as error:
            raise LadderRefusal(str(error)) from None
        count = row.get("members")
        if count is None and len(entries) == 1:
            count = DEFAULT_MEMBERS if members is None else members
        if type(count) is not int or count < 1:
            raise LadderRefusal(f"source {token!r} needs a positive whole member count")
        if any(item["source"] == token for item in result):
            raise LadderRefusal(f"source family {token!r} is listed more than once")
        result.append({"source": token, "members": count})
    total = sum(row["members"] for row in result)
    if members is not None and total != members:
        raise LadderRefusal(f"source counts sum to {total} members, but --members/[ensemble].members is {members}")
    return tuple(result)


def source_spec_argument(value):
    """Argparse validation independent of the separately parsed --members flag."""
    try:
        parse_source_spec(value)
    except ValueError as error:
        from argparse import ArgumentTypeError
        raise ArgumentTypeError(str(error)) from None
    return value


def _owner_call(function, **kwargs):
    """Pass model/count context to owners that declare it; retain older owners."""
    from inspect import signature, Parameter
    params = signature(function).parameters
    if not any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return function(**kwargs)


@dataclass(frozen=True)
class WofsOptions:
    members: int = DEFAULT_MEMBERS
    wind_amplitude: float = DEFAULT_WIND_AMPLITUDE
    physics: str = DEFAULT_PHYSICS
    surface: bool = True
    max_lag_hours: int = DEFAULT_MAX_LAG_HOURS
    base_seed: int = DEFAULT_SEED
    #: The source archive's provider choice, read by the source's selection owner.
    provider: str | None = None
    family_weights: dict | str | None = None
    family_providers: dict = field(default_factory=dict)
    arms: tuple = PHYSICS_ARMS

    def __post_init__(self):
        from woof.ensemble.lineage import family_weights
        family_weights(self.family_weights)
        if isinstance(self.members, bool) or not isinstance(self.members, int) or self.members < 1:
            raise LadderRefusal("--members must be a positive whole number")
        if (isinstance(self.wind_amplitude, bool) or not isinstance(self.wind_amplitude, (int, float))
                or not 0.0 <= float(self.wind_amplitude) < 50.0):
            raise LadderRefusal("--wind-amplitude is m/s in [0, 50); 0 turns the wind offsets off")
        if self.physics not in PHYSICS_MODES:
            raise LadderRefusal(f"--physics-arms must be one of {', '.join(PHYSICS_MODES)}")
        if (isinstance(self.max_lag_hours, bool) or not isinstance(self.max_lag_hours, int)
                or self.max_lag_hours < 0):
            raise LadderRefusal("--max-lag-hours must be a whole number of hours, 0 or more")
        if not self.arms or self.arms[0][0] != "base" or self.arms[0][1]:
            raise LadderRefusal("the physics arm set must start with the config's own physics ('base', {})")


@dataclass(frozen=True)
class LadderMember:
    index: int
    rung: str
    source: str
    cycle: datetime
    source_member: str | None
    lag_hours: int
    physics_arm: str = "base"
    physics: dict = field(default_factory=dict)
    surface: dict | None = None
    wind_amplitude: float = 0.0
    seed: int = 0
    model: str | None = None
    parent_receipt: dict | str | None = None
    family: str = ""

    def trajectory(self) -> dict:
        row = {"source": self.source, "cycle": self.cycle.strftime("%Y-%m-%dT%H")}
        if self.source_member is not None:
            row["member"] = self.source_member
        return row

    def provenance(self) -> dict:
        """The member's label, TOML-safe (no null values)."""
        row = {"schema": "gpuwm-member-provenance.v1", "member": self.index, "rung": self.rung, "source": self.source,
               "cycle": self.cycle.strftime("%Y-%m-%dT%H"), "lag_hours": self.lag_hours,
               "physics_arm": self.physics_arm, "seed": self.seed,
               "family": self.family or self.source}
        if self.source_member is not None:
            row["source_member"] = self.source_member
        if self.physics:
            row["physics"] = dict(self.physics)
        if self.wind_amplitude > 0:
            row["perturbation"] = {
                "kind": "uniform-wind", "amplitude_m_s": float(self.wind_amplitude),
                "applied": "after initialized parent checkpoint restore, before the first branch step",
                "spread": "uncalibrated: one seeded constant U and V offset per member"}
        if self.surface:
            row["surface"] = dict(self.surface)
        if self.rung == RUNG_FILL:
            row["branch"] = {"kind": "initialized-checkpoint", "model_elapsed_seconds": 0}
        from woof.ensemble.lineage import build_lineage
        return build_lineage(row, model=self.model, receipt=self.parent_receipt)


from woof.ensemble.lineage import provenance_label


@dataclass(frozen=True)
class WofsLadder:
    source: str
    cycle: datetime
    start: datetime
    end: datetime
    per_member_sources: bool
    members: tuple
    options: WofsOptions
    gaps: tuple = ()

    def counts(self) -> dict:
        result = {RUNG_SOURCE: 0, RUNG_LAGGED: 0, RUNG_FILL: 0}
        for member in self.members:
            result[member.rung] += 1
        return result

    def provenance_records(self):
        from woof.ensemble.lineage import weight_records
        return weight_records([member.provenance() for member in self.members], self.options.family_weights)
    def family_counts(self):
        result = {}
        for member in self.members:
            family = member.family or member.source
            counts = result.setdefault(family, {RUNG_SOURCE: 0, RUNG_LAGGED: 0, RUNG_FILL: 0})
            counts[member.rung] += 1
        return result

    def source_specs(self):
        return [{"source": family, "members": sum(counts.values())}
                for family, counts in self.family_counts().items()]

    def describe(self) -> dict:
        return {"contract": CONTRACT, "source": self.source,
                "cycle": self.cycle.isoformat(), "start": self.start.isoformat(),
                "end": self.end.isoformat(), "per_member_sources": self.per_member_sources,
                "rungs": self.counts(), "families": self.family_counts(), "gaps": list(self.gaps),
                "options": {"members": self.options.members,
                            "wind_amplitude_m_s": float(self.options.wind_amplitude),
                            "physics_arms": self.options.physics,
                            "surface_fill": self.options.surface,
                            "max_lag_hours": self.options.max_lag_hours,
                            "base_seed": self.options.base_seed},
                "physics_arm_set": [{"name": name, "physics": dict(physics)}
                                    for name, physics in self.options.arms],
                "family_weights": self.provenance_records()[1],
                "members": list(self.provenance_records()[0])}

    def ensemble_table(self, existing=None) -> dict:
        """The ``[ensemble]`` table that runs this ladder on an existing door."""
        table = {}
        if isinstance(existing, dict):
            for key in ("keep_member_files", "thresholds", "max_ordinary_members_per_device",
                        "base_seed", "retain_member_diagnostics", "member_device_ids", "stochastic",
                        "hydro", "sounding_points"):
                if key in existing:
                    table[key] = existing[key]
        table["sources"] = self.source_specs()
        table["members"] = len(self.members)
        table["provenance"], table["family_weights"] = self.provenance_records()
        table["provenance"] = list(table["provenance"])
        table["recipe"] = "wofs"
        table["base_seed"] = self.options.base_seed
        table["keep_member_files"] = table.get("keep_member_files", True)
        table["retain_member_diagnostics"] = True
        table["trajectories"] = [member.trajectory() for member in self.members]
        table["member_variants"] = [
            {"name": f"m{member.index:04d}-{member.physics_arm}",
             "physics": dict(member.physics), "surface": dict(member.surface or {}),
             "wind": float(member.wind_amplitude)} for member in self.members]
        return table


def _unit(seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{CONTRACT}:{seed}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def surface_draw(seed: int) -> dict:
    """A member's seeded surface-state values, rounded so they survive FP32 and TOML."""
    low, high = SOIL_MOISTURE_RANGE
    sms = round(low + (high - low) * _unit(seed, "soil_moisture_scale"), 4)
    low, high = SST_OFFSET_RANGE_K
    sst = round(low + (high - low) * _unit(seed, "sst_offset_k"), 3)
    return {"soil_moisture_scale": sms, "sst_offset_k": sst}


def selection_owner(adapter):
    """The module the adapter row names as the source's member-selection owner, or None.

    The table law: a source whose members are not a packaged member set
    (an analysis ensemble served by one provider with a key) names the
    module that owns the fact, and the generic line asks that module.
    """
    if not adapter.selection_owner:
        return None
    from importlib import import_module
    return import_module(adapter.selection_owner)


def source_population(source, cycle, *, provider=None, members=None, base_seed=0, index_offset=0):
    """Rung 1: the source's own (or declared) ensemble at ``cycle``, else its one trajectory."""
    from woof.ensemble.recipes import SourceTrajectory, ensemble_population
    from woof.source_adapters import get_source_adapter, canonical_source_token
    adapter = get_source_adapter(source)
    token = canonical_source_token(source)
    owner = selection_owner(adapter)
    if owner is not None:
        selected = _owner_call(owner.ensemble_members, provider=provider, source=token,
                               members=members, base_seed=base_seed, index_offset=index_offset,
                               cycle=cycle)
        if selected:
            return tuple(SourceTrajectory(token, cycle, member) for member in selected)
    ensemble = adapter.source_id if adapter.member_set else None
    if ensemble is not None:
        try:
            if get_source_adapter(ensemble).runnable:
                return ensemble_population(ensemble, cycle)
        except ValueError:
            pass
    return (SourceTrajectory(token, cycle),)


def _covers(trajectory, start, end):
    try:
        trajectory.window(start, end, native_bracketing=True)
    except ValueError:
        return False
    return True


def plan_ladder(*, source, cycle, start, end, options=None, per_member_sources=True,
                population=None, index_offset=0, family=None) -> WofsLadder:
    """Fill ``options.members`` members down the ladder; no download, no CUDA.

    ``per_member_sources`` records whether the prepared-chain door is used.
    Config-driven ERA5 members bind their own forcing through the runtime provider.  ``population`` replaces the
    source-table lookup (a callable ``(source, cycle) -> trajectories``),
    for tests and for doors that already know the members.
    """
    from woof.ensemble.seeds import member_seed
    from woof.source_adapters import get_source_adapter, canonical_source_token
    from woof.source_cycles import cycle_grid_for
    options = options or WofsOptions()
    population = population or (lambda source, cycle: source_population(
        source, cycle, provider=options.provider, members=options.members,
        base_seed=options.base_seed, index_offset=index_offset))
    count = options.members
    gaps = []
    adapter = get_source_adapter(source)
    source = canonical_source_token(source)
    family = family or source
    real = []                       # (trajectory, rung, lag)
    for item in population(source, cycle):
        if _covers(item, start, end):
            real.append((item, RUNG_SOURCE, 0))
            if len(real) >= count:
                break
    if not real:
        raise LadderRefusal(f"{adapter.source_id} {cycle:%Y-%m-%dT%H}Z does not cover the requested window")
    grid = cycle_grid_for(adapter.source_id)
    if adapter.time_axis != "analysis_times" and grid is not None:
        for lag in range(1, options.max_lag_hours + 1):
            if len(real) >= count:
                break
            earlier = cycle - timedelta(hours=lag)
            if earlier.hour not in grid.hours:
                continue
            for item in population(source, earlier):
                if _covers(item, start, end):
                    real.append((item, RUNG_LAGGED, lag))
                    if len(real) >= count:
                        break
    elif adapter.time_axis == "analysis_times":
        gaps.append(f"{adapter.source_id} publishes analyses, so it has no time-lagged forecasts")
    members = []
    arms = options.arms
    for index, (item, rung, lag) in enumerate(real[:count]):
        index += index_offset
        seed = member_seed(options.base_seed, index)
        arm = arms[(index - index_offset) % len(arms)] if options.physics == "all" else arms[0]
        members.append(LadderMember(index, rung, item.source, item.cycle, item.member, lag,
                                    physics_arm=arm[0], physics=dict(arm[1]), seed=seed, family=family))
    fill = count - len(members)
    if fill > 0:
        if options.wind_amplitude <= 0 and not options.surface and options.physics == "off":
            raise LadderRefusal(
                f"{len(members)} distinct source trajectories cover this window and "
                f"{count} members were asked; with the wind offsets, the surface draws and the "
                "physics arms all off, the extra members would be copies of real ones "
                "(zero spread). Next: leave one of them on, or ask for fewer members")
        physics_arms = arms if options.physics != "off" else arms[:1]
        for k in range(fill):
            index = index_offset + len(members)
            item, _rung, lag = real[k % len(real)]
            seed = member_seed(options.base_seed, index)
            arm = physics_arms[(k + 1) % len(physics_arms)] if len(physics_arms) > 1 else physics_arms[0]
            members.append(LadderMember(
                index, RUNG_FILL, item.source, item.cycle, item.member, lag,
                physics_arm=arm[0], physics=dict(arm[1]),
                surface=surface_draw(seed) if options.surface else None,
                wind_amplitude=float(options.wind_amplitude), seed=seed, family=family))
    return WofsLadder(source, cycle, start, end, per_member_sources, tuple(members), options,
                      tuple(gaps))


def plan_families(*, sources, cycle, start, end, options=None, population=None):
    """Each family owns its ladder and physics arms; indices and seeds are global."""
    from woof.source_adapters import get_source_adapter
    specs = parse_source_spec(sources)
    count = sum(row["members"] for row in specs)
    options = options or WofsOptions(members=count)
    if options.members != count:
        raise LadderRefusal(f"source counts sum to {count} members, but ensemble members is {options.members}")
    members, gaps = [], []
    for row in specs:
        adapter = get_source_adapter(row["source"])
        arm_table = (adapter.model_row or {}).get("physics_arms")
        arms = (adapter.ensemble_physics_arms or options.arms) if arm_table is None else tuple(
            (arm["name"], arm["physics"]) for arm in arm_table)
        owner = selection_owner(adapter)
        provider = options.provider
        if owner is not None:
            provider = options.family_providers.get(row["source"],
                owner.provider_of({"source": row["source"],
                    **({"model": adapter.model_key} if adapter.model_key else {})}))
        ladder = plan_ladder(source=row["source"], cycle=cycle, start=start, end=end,
            options=replace(options, members=row["members"], arms=arms, provider=provider),
            population=population, index_offset=len(members), family=row["source"])
        members.extend(ladder.members)
        gaps.extend(ladder.gaps)
    return WofsLadder(",".join(row["source"] for row in specs), cycle, start, end,
                      all(per_member_route(row["source"]) for row in specs),
                      tuple(members), options, tuple(gaps))


def per_member_route(source) -> bool:
    """Does this source fetch and prepare each member on its own chain?"""
    from woof.regional_preparation import member_preparation_chain
    try:
        member_preparation_chain(str(source))
    except Exception:   # noqa: BLE001 - PlanError or an unregistered source: one input
        return False
    return True


def _utc(value):
    from woof.ensemble.recipes import trajectory_time
    return trajectory_time(value)


def ladder_for_payload(raw, options) -> WofsLadder:
    """The ladder for a config written by ``woof domain``."""
    fetch = raw.get("fetch")
    if not isinstance(fetch, dict) or not {"source", "cycle"} <= fetch.keys():
        raise LadderRefusal("the WoFS recipe plans members from the config's [fetch] source "
                            "and cycle, and this config has neither. Next: woof domain --help")
    experiment = raw.get("experiment") or {}
    start = _utc(str(experiment.get("start_time")).replace(" ", "T"))
    end = start + timedelta(seconds=float(experiment.get("run_seconds", 0)))
    cycle = _utc(str(fetch["cycle"]))
    source = str(fetch["source"])
    table = raw.get("ensemble") or {}
    if options.family_weights is None and table.get("family_weights") is not None:
        options = replace(options, family_weights=table["family_weights"])
    if table.get("sources"):
        specs = parse_source_spec(table["sources"], options.members)
        return plan_families(sources=specs, cycle=cycle, start=start, end=end, options=options)
    per_member = per_member_route(source) and "case_data" not in raw
    if options.provider is None:
        from woof.source_adapters import get_source_adapter
        owner = selection_owner(get_source_adapter(source))
        if owner is not None:
            options = replace(options, provider=owner.provider_of(fetch))
    return plan_ladder(source=source, cycle=cycle, start=start, end=end, options=options,
                       per_member_sources=per_member)


def expand_config(config, options, *, into=None):
    """Write a copy of ``config`` whose ``[ensemble]`` table runs the ladder.

    The copy sits beside the original (relative paths in it stay valid),
    with the companion WPS namelist copied under its stem.  Returns the new
    path and the ladder.
    """
    from pathlib import Path
    import shutil
    import tomllib

    from woof.toml_document import emit_experiment_toml

    config = Path(config)
    raw = tomllib.loads(config.read_text(encoding="utf-8-sig"))
    ladder = ladder_for_payload(raw, options)
    changed = dict(raw)
    changed["ensemble"] = ladder.ensemble_table(raw.get("ensemble"))
    from woof.experiment import load_experiment
    from woof.source_coverage import config_source_coverage_refusal
    experiment = load_experiment(config)
    for family in ladder.source_specs():
        refusal = config_source_coverage_refusal(experiment, family["source"])
        if refusal:
            raise LadderRefusal(f"family {family['source']}: {refusal}")
    # Bind every member to its own source chain and selection owner.
    from woof.ensemble.recipe_door import plan_recipe, member_fetch
    from woof.ensemble.request import EnsembleRequest
    request = EnsembleRequest.from_mapping(changed["ensemble"])
    recipe = plan_recipe(request, raw, experiment)
    fetches = []
    from woof.source_adapters import get_source_adapter
    batches = {}
    for member, planned in zip(ladder.members, recipe.members):
        hints = member_fetch(raw, experiment, recipe, planned)
        adapter = get_source_adapter(member.source)
        owner = selection_owner(adapter)
        provider = options.family_providers.get(member.family, adapter.model_key)
        if adapter.selection_provider_hint and provider is not None:
            hints[adapter.selection_provider_hint] = provider
        if owner is not None and member.source_member is not None:
            # Each model/cycle owns one shared batch. Distinct family batches
            # must never reuse another model's roster or output directory.
            family_key = hashlib.sha256(member.family.encode()).hexdigest()[:12]
            root = (config.parent / str(raw["fetch"].get("out", "data")) /
                    family_key / member.cycle.strftime("%Y%m%dT%H%M%S")).resolve()
            hints = owner.member_fetch_hints(hints, member.source_member,
                window_seconds=(ladder.end - member.cycle).total_seconds(), root=root)
            batches.setdefault((member.family, member.cycle), (owner, []))[1].append(member.index)
        fetches.append(hints)
    for owner, indices in batches.values():
        if hasattr(owner, "batch_fetch_hints"):
            grouped = owner.batch_fetch_hints([fetches[index] for index in indices])
            if len(grouped) != len(indices):
                raise LadderRefusal("selection owner changed the planned member count")
            for index, hints in zip(indices, grouped):
                fetches[index] = hints
    if batches or len(ladder.source_specs()) > 1:
        changed["ensemble"]["member_fetches"] = fetches
    # The config-driven provider applies only when all members share that route.
    if "case_data" in raw and len(ladder.source_specs()) == 1:
        adapter = get_source_adapter(ladder.members[0].source)
        owner = selection_owner(adapter)
        if owner is not None and ladder.members[0].source_member is not None:
            from pathlib import Path
            name, interval = owner.member_forcing()
            interval = adapter.forcing_interval_seconds or interval
            changed["fetch"] = fetches[0]
            changed["case_data"] = dict(raw["case_data"], forcing=[str(Path(fetches[0]["out"]) / name)],
                                        forcing_interval_s=interval)
    elif len(ladder.source_specs()) > 1:
        changed.pop("case_data", None)

    changed.setdefault("shared", {}).setdefault("nwp_diagnostics", 1)
    target = Path(into) if into is not None else config.with_name(
        f"{config.stem}.wofs{len(ladder.members)}.toml")
    target.write_text(emit_experiment_toml(changed), encoding="utf-8")
    companion = config.with_name(config.stem + ".namelist.wps")
    if companion.is_file():
        shutil.copyfile(companion, target.with_name(target.stem + ".namelist.wps"))
    (target.with_name(target.stem + ".ladder.json")).write_text(
        json.dumps(ladder.describe(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target, ladder


def summary_lines(ladder) -> list:
    counts = ladder.counts()
    lines = [f"wofs ladder: {len(ladder.members)} members from {ladder.source} "
             f"{ladder.cycle:%Y-%m-%dT%H}Z: {counts[RUNG_SOURCE]} source, "
             f"{counts[RUNG_LAGGED]} time-lagged, {counts[RUNG_FILL]} perturbation fill "
             "(uncalibrated spread; every member's provenance is in the receipt)"]
    lines.extend(f"wofs lineage: m{row['member']:04d} {row['parent_family']} {row['model_key']} family_weight={row['family_weight']:g}" for row in ladder.provenance_records()[0])
    lines.extend(f"wofs family {family}: {sum(counts.values())} members: "
                 f"{counts[RUNG_SOURCE]} source, {counts[RUNG_LAGGED]} time-lagged, "
                 f"{counts[RUNG_FILL]} perturbation fill"
                 for family, counts in ladder.family_counts().items())
    lines.extend(f"wofs ladder: note: {gap}" for gap in ladder.gaps)
    return lines


__all__ = ["CONTRACT", "DEFAULT_MEMBERS", "PHYSICS_ARMS", "LadderRefusal", "WofsOptions",
           "WofsLadder", "LadderMember", "plan_ladder", "surface_draw", "ladder_for_payload",
           "expand_config", "per_member_route", "selection_owner", "source_population",
           "summary_lines"]


def _family_namespace(args, source):
    """Owners see one source token even when the CLI carries a mixture."""
    from copy import copy
    from woof.source_adapters import get_source_adapter
    context = copy(args)
    context.source = source
    model = get_source_adapter(source).model_key
    if model is not None:
        context.model = model
    return context


def prepare_launch(args):
    """Materialize the ordinary domain config before go reads its request.

    Every flag is read the way :func:`woof.go_cli.go_main` reads its own:
    absent means the parser's default.  The hosted front doors and the
    ``ensemble`` alias hand go namespaces that carry only what they set,
    and a namespace from another command never carries these flags.

    A bare member count takes the ladder only where a ladder can be
    planned, a config with a ``[fetch]`` source and cycle.  Any other
    config keeps go's own member door, which refuses a one-input config
    by name instead of this planner refusing it for want of a table.
    """
    from pathlib import Path
    import argparse
    import sys
    import tomllib
    import uuid

    from woof.source_adapters import get_source_adapter

    config = getattr(args, "config", None)
    selected = getattr(args, "recipe", None) == "wofs"
    if config is None:
        if not all(getattr(args, key, None) for key in ("source", "init", "center")):
            raise LadderRefusal("supply CONFIG, or --source, --init and --center to create a WoFS-shape domain")
        selected = True
        from woof import domain_wizard
        specs = parse_source_spec(args.source, getattr(args, "members", None))
        args.members = sum(row["members"] for row in specs)
        first_source = specs[0]["source"]
        owner = selection_owner(get_source_adapter(first_source))
        provider_argv = [] if owner is None else owner.wizard_arguments(_family_namespace(args, first_source))
        hours = getattr(args, "hours", DEFAULT_HOURS)
        extent_km = getattr(args, "extent_km", DEFAULT_EXTENT_KM)
        geometry_key = hashlib.sha256(json.dumps(
            {"source": args.source, "init": args.init, "center": args.center, "hours": hours,
             "extent_km": extent_km, "card": getattr(args, "card", None),
             "geog_root": getattr(args, "geog_root", None), "provider": provider_argv},
            sort_keys=True, default=str).encode()).hexdigest()[:12]
        root = (Path(getattr(args, "outdir", None) or "ensemble-wofs").resolve()
                / "configurations" / geometry_key)
        root.mkdir(parents=True, exist_ok=True)
        config = root / "ensemble.toml"
        parser = argparse.ArgumentParser()
        domain_wizard.register_cli(parser.add_subparsers(dest="command"))
        argv = ["domain", "--point=" + args.center, "--source", first_source,
                "--cycle", args.init, "--hours", str(hours), "--root-dx", "3",
                "--point-extent-km", str(extent_km), "--history-interval", "900",
                "--nz", "49", "--out", str(config), *provider_argv]
        if getattr(args, "card", None):
            argv += ["--card", args.card]
        if getattr(args, "geog_root", None):
            argv += ["--geog-root", str(args.geog_root)]
        code = domain_wizard.domain_main(parser.parse_args(argv))
        if code:
            raise LadderRefusal(f"domain creation stopped with exit {code}")
        from woof.toml_document import emit_experiment_toml
        created = tomllib.loads(config.read_text(encoding="utf-8"))
        created["ensemble"] = {"recipe": "wofs", "members": args.members, "sources": list(specs)}
        config.write_text(emit_experiment_toml(created), encoding="utf-8")
    elif any(getattr(args, key, None) for key in ("source", "init", "center")):
        raise LadderRefusal("CONFIG already owns its source, time and domain; use --cycle to re-time it")
    config = Path(config)
    if not config.is_file():
        args.config = config
        return
    raw = tomllib.loads(config.read_text(encoding="utf-8-sig"))
    table = raw.get("ensemble") or {}
    selected = selected or isinstance(table, dict) and (table.get("recipe") == "wofs" or bool(table.get("sources")))
    fetch = raw.get("fetch")
    plannable = isinstance(fetch, dict) and {"source", "cycle"} <= fetch.keys()
    # A bare count takes the ladder unless an authored mechanism already
    # owns diversity. Explicit recipes and perturbation knobs retain priority.
    if (getattr(args, "recipe", None) is None and not table and plannable
            and getattr(args, "trajectories", None) is None
            and (getattr(args, "members", None) is not None
                 or getattr(args, "command", None) == "ensemble")):
        selected = True
    if not selected:
        args.config = config
        return
    count = getattr(args, "members", None)
    if table.get("sources"):
        specs = parse_source_spec(table["sources"], count if count is not None else table.get("members"))
        count = sum(row["members"] for row in specs)
    if count is None:
        count = table.get("members", DEFAULT_MEMBERS)
    owner = selection_owner(get_source_adapter(str(fetch["source"]))) if plannable else None
    family_providers = {}
    for spec in table.get("sources", [{"source": fetch["source"]}] if plannable else []):
        token = spec["source"]
        selection = selection_owner(get_source_adapter(token))
        if selection is not None:
            context = _family_namespace(args, token)
            provider = selection.provider_flag(context)
            if provider is None and token == fetch["source"]:
                provider = selection.provider_of(fetch)
            family_providers[token] = provider
    options = WofsOptions(
        members=count, family_providers=family_providers,
        family_weights=getattr(args, "family_weights", None),
        wind_amplitude=getattr(args, "wind_amplitude", DEFAULT_WIND_AMPLITUDE),
        physics=getattr(args, "physics_arms", DEFAULT_PHYSICS),
        surface=not getattr(args, "no_surface_fill", False),
        max_lag_hours=getattr(args, "max_lag_hours", DEFAULT_MAX_LAG_HOURS),
        base_seed=getattr(args, "ensemble_seed", DEFAULT_SEED),
        provider=family_providers.get(str(fetch["source"])) if plannable else None)
    # Apply --cycle before planning so labels name the bytes actually fetched.
    if getattr(args, "cycle", None) is not None:
        from woof.go_cli import _at_flag_cycle
        config, raw = _at_flag_cycle(args, config, raw, args.cycle)
        args.cycle = None
    target = config.with_name(config.stem + ".wofs-" + uuid.uuid4().hex[:10] + ".toml")
    args.config, ladder = expand_config(config, options, into=target)
    args.recipe = None
    # Under --readiness the readiness document owns stdout.
    stream = sys.stderr if getattr(args, "readiness", False) else sys.stdout
    for line in summary_lines(ladder):
        print(line, file=stream, flush=True)


def runtime_member_provider(request):
    """Bind ordinary config-driven members to their physics and surface arms."""
    acquired = {}
    def provider(*, shared_inputs, member_id, request):
        from dataclasses import replace
        variant = request.member_variants[member_id]
        experiment = shared_inputs.experiment
        domains = tuple(replace(domain, run=replace(domain.run, **variant["physics"]))
                        for domain in experiment.domains)
        data = shared_inputs.case_data
        if request.member_fetches:
            from pathlib import Path
            import subprocess
            import sys
            from woof.runplan import _fetch_arguments_from_hints
            from woof.cli import _join_negative_coordinates
            from woof.source_adapters import get_source_adapter
            hints = request.member_fetches[member_id]
            key = hints["out"]
            adapter = get_source_adapter(str(hints["source"]))
            name, interval = selection_owner(adapter).member_forcing()
            interval = adapter.forcing_interval_seconds or interval
            if key not in acquired:
                arguments = _fetch_arguments_from_hints(hints, out=Path(key))
                # Only a source whose default fetch writes a request template
                # takes --retrieve (the adapter row says so); woof fetch
                # refuses it for every other source. Breakage it prevents: a
                # one-family generated ensemble (--source earth2-fcn3 --members N)
                # stopped at its first member fetch with "--retrieve applies only
                # to sources whose default fetch writes a request template".
                if adapter.fetch_requires_retrieve and "--retrieve" not in arguments:
                    arguments.append("--retrieve")
                subprocess.run([sys.executable, "-m", "woof", "fetch",
                                *_join_negative_coordinates(arguments)], check=True)
                acquired[key] = Path(key) / name
            data = replace(data, forcing=(acquired[key],), forcing_interval_s=interval)
        return replace(shared_inputs, experiment=replace(experiment, domains=domains), case_data=data)
    return provider


def prewarm_member_kernels(experiments):
    """Submit every physics arm to the existing all-core compile pool once.

    A single forecast's static plan covers only its own selectors. Waiting
    for later ensemble arms to reach step one compiled their new schemes
    serially, despite the first arm already having started a kernel plan.
    """
    from woof import kernel_plan, kernel_prewarm
    from woof.local_gpu import no_local_gpu
    if no_local_gpu() or not kernel_plan.enabled():
        return
    experiments = tuple(experiments)
    tag = "wofs|" + "|".join(sorted({kernel_prewarm.plan_tag(exp) for exp in experiments}))
    if not kernel_plan.install(tag_text=tag):
        return
    kernel_plan.start()
    seen = set()
    for experiment in experiments:
        for label, loader, arguments in kernel_prewarm.static_jobs(experiment):
            identity = (label, repr(arguments))
            if identity not in seen:
                seen.add(identity)
                kernel_plan.submit(kernel_plan.ahead, loader, *arguments)
    kernel_plan.close()
