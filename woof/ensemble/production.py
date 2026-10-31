"""Shared prepared inputs and exact ordinary-member ensemble execution.

Preparation belongs to the caller and is performed once. Qualified native
batch executors can replace one admitted pack through an explicit callback.
The ordinary fallback retains the original runner, clock and physics.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from collections.abc import Mapping
import json
import hashlib
import inspect
from pathlib import Path
import re
import time
from datetime import timedelta

from woof.ensemble.execution import (
    MemberRunControl, execute_member_packing, failed_member_rows, member_run_scope,
    name_failing_members, execute_concurrent_members,
)
from woof.ensemble.request import EnsembleRequest
from woof.ensemble.runtime_context import MemberOutputCapture, member_output_scope
from woof.ensemble.packing import CardBudget, MemberBatch, pack_members
from woof.ensemble.radar_output import radar_enabled, member_history_required, finish_member_radar


def _atomic_json(path, record):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_scalar) + "\n", encoding="utf-8")
    temporary.replace(path)


def release_finished_member(array_module=None):
    """Free a finished ordinary member's device state before the next starts.

    DomainState and PhysicsDriver reference each other, so a finished
    member's arrays are cyclic garbage that only a collector pass frees.
    Breakage this prevents: packing prices one live member per card and the
    ordinary runner's own admission counts only driver-free plus pool-free
    bytes, so each uncollected member read as unavailable memory. Measured
    on a 150x150x49 case, the default pool grew 591 MB per finished member;
    on a domain needing more than half the card the third member would be
    refused or dropped to tiles after two full forecasts.
    """
    import gc
    # The frees above are queued on this thread's stream. An array module
    # without a stream interface (a planning fixture) has nothing to wait for.
    current_stream = getattr(getattr(array_module, "cuda", None), "get_current_stream", None)
    if current_stream is not None:
        current_stream().synchronize()
    gc.collect()


@contextmanager
def finished_member_release(array_module=None):
    """Release on the way out of one ordinary member's forecast."""
    try:
        yield
    except BaseException:
        # The failing member's frames still hold its state. Collect what is
        # already free and keep the first error as the one that is raised.
        release_finished_member()
        raise
    release_finished_member(array_module)


def _json_scalar(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    from datetime import datetime
    if isinstance(value, datetime):
        return value.isoformat()
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"ensemble receipt cannot serialize {type(value).__name__}")


# Allocation reuse changes no experiment selector. Its qualification covers
# the complete original carried fields and original prepared runner owners.
_COMPONENT_WORKSPACE_REUSE_QUALIFIED = True
_COMPONENT_EDGE_STATE_ONLY_QUALIFIED = True


def _component_route_receipt(route):
    """Freeze counters and final clock metadata without retaining live owners."""
    from dataclasses import fields
    from fractions import Fraction
    from woof.core.clock import DomainClock
    def scalar(value):
        if isinstance(value, DomainClock):
            return {name: getattr(value, name) for name in value.__slots__}
        if is_dataclass(value):
            return {field.name: getattr(value, field.name) for field in fields(value)}
        if isinstance(value, Fraction):
            return {"numerator": value.numerator, "denominator": value.denominator}
        return _json_scalar(value)
    return json.loads(json.dumps(route, default=scalar))


def _prepared_component_route_plan(member_inputs, *, card, collector, array_module,
                                   device_profile=None, stochastic_provider=None):
    """Price a genuine parent/nest component wave before any runner starts."""
    from fractions import Fraction
    from woof.core.preflight import estimate_experiment, FORECAST_POOL_HEADROOM
    from woof.core.mynn_pbl_scratch import mynn_pricing_memory
    from woof.ensemble.admission import ordinary_memory_model_from_estimate
    from woof.ensemble.prepared_component_reservation import plan_prepared_component_reservation
    from woof.ensemble.production_memory import ordinary_ensemble_memory_model
    from woof.ensemble.optional_routing import enabled_for
    if any(enabled_for(value) for value in member_inputs.values()):
        return None, "NextGen requires the original resident hydro driver for each member"
    if stochastic_provider is not None and any(stochastic_provider.enabled_for_experiment(value.experiment)
                                               for value in member_inputs.values()):
        return None, "active stochastic rates require their original per-member driver callbacks; packed production has no stochastic leaf handoff"
    estimates = []
    forecast_bytes = {}
    try:
        total = card.total_bytes
        if total is None:
            _, total = array_module.cuda.runtime.memGetInfo()
        with mynn_pricing_memory(total_bytes=int(total), free_bytes=card.available_bytes):
            for member, prepared in member_inputs.items():
                estimate = estimate_experiment(prepared.experiment,
                    forcing_interval_seconds=prepared.boundary_interval_seconds,
                    profile=device_profile, vram_gib=int(total) / (1024 ** 3))
                estimates.append((estimate, prepared.experiment))
                forecast_bytes[member] = ordinary_memory_model_from_estimate(estimate,
                    inventory_id="original-forecast-before-component-runners").required_bytes(1)
            first, experiment = estimates[0]
            _, external, output_margin = ordinary_ensemble_memory_model(first, experiment, collector,
                inventory_id="original-output-before-component-runners", other_variants=tuple(estimates[1:]))
            rows = tuple(component.inventory(1) for component in external)
            output_bytes = sum(row["required_bytes"] for row in rows) + output_margin.required_bytes(rows)
            extra_queues = 16 << 20
            fraction = Fraction(str(FORECAST_POOL_HEADROOM)) - 1
            queue_margin = (extra_queues * fraction.numerator + fraction.denominator - 1) // fraction.denominator
            decision = plan_prepared_component_reservation(member_inputs,
                ordinary_forecast_bytes=forecast_bytes,
                fixed_bytes={"collector": output_bytes, "stochastic": 0,
                             "cuda_owners": extra_queues, "allocator_margin": queue_margin},
                evidence={"ordinary": "complete original per-member estimate on the selected card before runner threads",
                          "native": "automatic immutable prepared authorities and original component inventories",
                          "collector": "original complete headline output reservations and allocator margin",
                          "stochastic": "inactive for this component cohort",
                          "cuda_owners": "16 MiB additional route/RPC/event ownership reserve",
                          "allocator_margin": "original forecast headroom applied to the additional queue reserve"},
                available_bytes=card.available_bytes, device_id=card.device_id,
                reuse_original_workspaces=_COMPONENT_WORKSPACE_REUSE_QUALIFIED,
                edge_state_only=_COMPONENT_EDGE_STATE_ONLY_QUALIFIED)
    except (ValueError, TypeError, AttributeError, KeyError, OSError) as error:
        return None, f"the prepared component allocation inventory cannot bind its original authorities: {error}"
    return (decision, None) if decision.eligible else (None, decision.ordinary_reason)


class PreparedEnsembleSession:
    """The worker and source-provider interface, independent of a CLI route.

    Input providers receive immutable shared preparation and a global member
    index. They return independently bound prepared inputs, including boundary
    source authority. No source/perturbation descriptor is silently ignored.

    Every member of an N > 1 session runs its own inputs: a member roster,
    an input provider, a source execution or active stochastic physics.
    A session with none of them refuses at first use, because it would run
    N copies of one forecast and publish zero spread and 0 or 1
    probabilities.  ``identical_members`` is the one exception and no front
    door sets it: an engine gate that runs N copies on purpose, to compare
    each with the ordinary forecast word for word, states that purpose
    there, and the run's manifest records it.
    """
    def __init__(self, request, *, output_directory, collector=None,
                 input_provider=None, native_executor=None, memory_model=None,
                 cards=None, device_scope=None, renderer=None, member_roster=None,
                 stochastic_provider=None, array_module=None,
                 card_bootstrap_factory=None, source_execution=None,
                 identical_members=None, restart_roster=None):
        self.request = EnsembleRequest.from_mapping(request)
        if identical_members is not None and (
                not isinstance(identical_members, str) or not identical_members.strip()):
            raise ValueError("identical_members states, in words, why this engine gate "
                             "runs copies of one forecast")
        self.identical_members = identical_members
        self.restart_roster = None if restart_roster is None else Path(restart_roster)
        self.output_directory = Path(output_directory)
        self.collector = collector
        self.input_provider = input_provider
        self.native_executor = native_executor
        self.memory_model = memory_model
        self.cards = cards
        self.device_scope = device_scope
        self.renderer = renderer
        self.member_roster = member_roster
        self.stochastic_provider = stochastic_provider
        self.array_module = array_module
        self.card_bootstrap_factory = card_bootstrap_factory
        self.source_execution = source_execution
        if source_execution is not None:
            from woof.ensemble.posted_execution import PostedRecipeExecution
            from woof.ensemble.ordinary_execution import OrdinaryRecipeExecution
            if not isinstance(source_execution, (PostedRecipeExecution, OrdinaryRecipeExecution)):
                raise TypeError("recipe source execution needs its concrete ordinary or changed-field preparation owner")
            if input_provider is not None or member_roster is not None:
                raise ValueError("a posted source owner cannot also replace inputs through another member provider")
        self._stochastic_authorities = {}
        self._stochastic_input_bindings = {}
        self._surface_receipts = {}
        self._initial_receipts = {}
        self._member_land_layouts = {}
        if self.stochastic_provider is None and self.request.stochastic is not None:
            from woof.ensemble.stochastic_model import StochasticModelProvider
            self.stochastic_provider = StochasticModelProvider.from_mapping(self.request.stochastic)
        self.last_manifest = None
        self.last_output_directory = None
        self.member_order = (tuple(source_execution.member_order) if source_execution is not None else
            tuple(member.member_id for member in member_roster.members)
            if member_roster is not None else tuple(range(self.request.members)))
        if len(self.member_order) != self.request.members:
            raise ValueError("requested member count differs from the verified source roster")
        if (self.request.member_variants
                and self.request.recipe != "wofs"
                and input_provider is None and member_roster is None and source_execution is None):
            raise ValueError("member_variants selects each member's own land and surface state, "
                             "and this session has no bound member inputs: every member would "
                             "run one prepared land column under its variant's name. "
                             "Next: woof ensemble CONFIG --recipe member-roster")
        from woof.ensemble.initial_perturbation import is_initial_perturbation
        if ((self.request.sources or (self.request.perturbation is not None
                                    and not is_initial_perturbation(self.request.perturbation)))
                and input_provider is None and member_roster is None and source_execution is None):
            # Breakage it prevents: the request names a source (or a
            # perturbation) per member, and this session would run every
            # member on the one prepared input under those names.
            from woof.ensemble_admission import unbound_sources_refusal
            raise ValueError(unbound_sources_refusal())

    def _random_members(self, experiment):
        """Does stochastic physics make this session's members differ?"""
        from woof.ensemble.initial_perturbation import is_initial_perturbation
        if is_initial_perturbation(self.request.perturbation):
            return True
        selected = any(getattr(domain.run, f"spp_{name}", 0)
                       for domain in getattr(experiment, "domains", ()) or ()
                       for name in ("conv", "pbl", "lsm"))
        provider = self.stochastic_provider
        if provider is None:
            # Native selectors in the prepared configuration bind their own
            # provider in run_prepared.
            return selected
        declared = getattr(provider, "spp", False)
        return bool(getattr(provider, "sppt", None) is not None
                    or getattr(provider, "skebs_psi", None) is not None or selected
                    or isinstance(declared, dict) and any(declared.values()))

    def _require_member_inputs(self, experiment=None):
        """Every member of an N > 1 session runs its own inputs.

        Breakage it prevents: a session with no member source hands every
        member the same prepared inputs.  The run is N copies of one
        forecast, published with zero spread and probabilities of 0 or 1.
        A recipe names real source trajectories, so it is refused in its
        own words when no door supplied them.
        """
        if (self.input_provider is not None or self.member_roster is not None
                or self.source_execution is not None):
            return
        if self.request.recipe is not None:
            raise ValueError(
                f"the {self.request.recipe} recipe needs each member's own prepared "
                "source, and this door prepared one trajectory. Next: woof ensemble "
                f"CONFIG --recipe {self.request.recipe} (or woof go / woof run)")
        if (self.request.members == 1 or self.identical_members is not None
                or self._random_members(experiment)):
            return
        from woof.ensemble_admission import one_input_refusal
        raise ValueError(one_input_refusal(
            self.request.members,
            "This ensemble session was given no member sources, so every member "
            "would run the same prepared inputs."))

    def run_prepared(self, runner, inputs, *, output_directory=None, **runner_options):
        from woof.ensemble.initial_perturbation import is_initial_perturbation
        experiment = getattr(inputs, "experiment", None)
        if self.stochastic_provider is not None:
            experiment = self.stochastic_provider.configure_experiment(experiment)
        self._require_member_inputs(experiment)
        started = time.perf_counter()
        started_monotonic = time.monotonic()
        out = Path(output_directory or self.output_directory)
        out.mkdir(parents=True, exist_ok=True)
        if self.request.recipe == "wofs":
            from woof.ensemble.checkpoint_fill import CheckpointFill
            self._checkpoint_fill = CheckpointFill(out / "parents")
        if self.request.sounding_points:
            from woof.soundings import binary
            binary()
        if (out / "ensemble-run.json").exists() and self.restart_roster is None:
            raise FileExistsError("ensemble output already exists; use a new run directory")
        exp = inputs.experiment
        if self.stochastic_provider is not None:
            inputs = self._configured_member_inputs(inputs)
            exp = inputs.experiment
            if self.member_roster is not None:
                from woof.ensemble.member_preparation import PreparedMemberRoster
                original_roster = self.member_roster
                self.member_roster = PreparedMemberRoster(original_roster.recipe,
                    tuple(replace(member, inputs=self._configured_member_inputs(member.inputs))
                          for member in original_roster.members),
                    shared_geometry_sha256=original_roster.geometry_sha256)
        # Bind each provider exactly once before sizing. Its own configuration,
        # initial state and complete boundary trajectory remain its authority.
        member_inputs_by_id = {}
        for member in self.member_order:
            prepared = (self.source_execution.planning_inputs(member) if self.source_execution is not None else
                self.member_roster.select((member,))[0].inputs
                if self.member_roster is not None else inputs if self.input_provider is None else
                self.input_provider(shared_inputs=inputs, member_id=member, request=self.request))
            member_inputs_by_id[member] = self._configured_member_inputs(prepared)
        # Native selectors belong to each original member configuration. An
        # automatic owner preserves off variants while owning active variants.
        if self.stochastic_provider is None and any(
                getattr(domain.run, f"spp_{name}", 0)
                for prepared in (inputs, *member_inputs_by_id.values())
                for domain in getattr(prepared.experiment, "domains", ())
                for name in ("conv", "pbl", "lsm")):
            from woof.ensemble.stochastic_model import StochasticModelProvider
            self.stochastic_provider = StochasticModelProvider()
            inputs = self._configured_member_inputs(inputs)
            exp = inputs.experiment
            member_inputs_by_id = {member: self._configured_member_inputs(prepared)
                                  for member, prepared in member_inputs_by_id.items()}
        from woof.ensemble.lineage import bind_runtime_stochastic
        self.request = bind_runtime_stochastic(self.request, {
            member: self._stochastic_authorities[id(prepared)]
            for member, prepared in member_inputs_by_id.items()
            if id(prepared) in self._stochastic_authorities})
        self._remember_member_land_layouts(member_inputs_by_id)
        from woof.ensemble.calibration_admission import random_warnings
        warnings = random_warnings(perturbation=self.request.perturbation,
                                   stochastic=self.request.stochastic)
        if (not warnings and self.stochastic_provider is not None and any(
                self.stochastic_provider.enabled_for_experiment(prepared.experiment)
                for prepared in member_inputs_by_id.values())):
            from woof.ensemble_admission import UNCALIBRATED_SPREAD_WARNING
            warnings = [UNCALIBRATED_SPREAD_WARNING]
        if warnings:
            import sys
            for warning in warnings:
                print("WARNING ensemble: " + warning, file=sys.stderr, flush=True)
        checkpoint_members = self.restart_roster is not None or any(
            float(getattr(prepared.experiment, "restart_interval_s", 0) or 0) > 0
            for prepared in member_inputs_by_id.values())
        resumed, resume_rows = None, {}
        if runner_options.get("restart") is not None:
            raise ValueError("an ensemble needs --restart-roster, because one checkpoint cannot identify every member")
        if checkpoint_members:
            from woof.ensemble import restart_roster
            identities = restart_roster.identity_record(self.request, self.member_order,
                member_inputs_by_id, {member: self._member_seed(member) for member in self.member_order})
            if self.restart_roster is not None:
                resumed = restart_roster.verify_roster(self.restart_roster, identities, root=out)
                resume_rows = {row["member_id"]: row for row in resumed["members"]}
                # The run kept writing after this roster sealed (the next
                # frame, the history ledger, checkpoint retirement); put the
                # sealed files back before the collector reads them.
                restored = restart_roster.restore_roster(resumed, root=out)
                if restored["restored"] or restored["removed_after_seal"]:
                    import sys
                    print(f"ensemble: resume restored {len(restored['restored'])} of "
                          f"{len(resumed['files'])} sealed file(s) the run changed after the roster "
                          f"seal and removed {len(restored['removed_after_seal'])} checkpoint or "
                          f"diagnostic file(s) written after it", file=sys.stderr, flush=True)
            restart_roster.atomic_json(out / restart_roster.IDENTITIES, identities)
            restart_roster.atomic_json(out / restart_roster.INPUTS, {"schema": restart_roster.CONTRACT,
                "members": [{"member_id": member, **{name: str(getattr(prepared, name, "") or "")
                    for name in ("prepared_root", "experiment_config", "wps_namelist")}}
                    for member, prepared in member_inputs_by_id.items()]})
        if self.collector is None:
            from woof import rustwx
            from woof.ensemble.batch_product_output import HeadlineDiagnosticCollector
            renderer = self.renderer or rustwx.find_renderer()
            if renderer is None:
                raise RuntimeError(rustwx.renderer_remedy())
            if self.source_execution is not None:
                metadata = tuple(self.source_execution.member_metadata.values())
            elif self.member_roster is not None:
                metadata = self.member_roster.receipts
            elif self.request.retain_member_diagnostics:
                from woof.ensemble.seeds import member_seed
                metadata = tuple({"member_id": member,
                    "seed": member_seed(self.request.base_seed, member),
                    "source": getattr(prepared, "source", None),
                    "prepared_authority_sha256": dict(getattr(prepared, "authority_sha256", {}) or {}),
                    "prepared_head_sha256": getattr(prepared, "prepared_head_sha256", None)}
                    for member, prepared in member_inputs_by_id.items())
            else:
                metadata = ()
            if self.request.provenance:
                previous = {row.get("member_id", row.get("member")): row for row in metadata}
                metadata = tuple({**previous.get(member, {}), **self.request.provenance[member],
                                  "member_id": member} for member in self.member_order)
            if self.request.trajectories and any("weight" in row for row in self.request.trajectories):
                previous = {row["member_id"]: row for row in metadata}
                metadata = tuple({**previous.get(member, {}), "member_id": member,
                    "weight": self.request.trajectories[position]["weight"],
                    "stratum": self.request.trajectories[position].get("stratum")}
                    for position, member in enumerate(self.member_order))
            self.collector = HeadlineDiagnosticCollector(out, members=self.request.members,
                renderer=renderer, start_time=exp.start_time,
                thresholds=self.request.thresholds, comparison=self.request.comparison,
                keep_member_files=self.request.keep_member_files,
                member_order=self.member_order,
                member_metadata=metadata,
                retain_member_diagnostics=self.request.retain_member_diagnostics,
                columns_end_time=exp.start_time + timedelta(seconds=float(exp.run_seconds)),
                **({"resume": True} if resumed is not None else {}))
        if self.request.cyclone_products and 'common_grid' in self.request.cyclone_products:
            from woof.tc_accumulator import validate_swath_request
            validate_swath_request(self.request.cyclone_products)
            if resumed is not None:
                raise ValueError('every-step ensemble swaths require an uninterrupted run; their diagnostic archive is not in the restart roster')
            grid_id = self.request.cyclone_products.get('grid_id', 2)
            if not any(domain.grid_id == grid_id for domain in exp.domains):
                raise ValueError(f'cyclone swath grid_id {grid_id} is absent from the member domains')
            self.collector.cyclone_request = self.request.cyclone_products
        from woof.ensemble.optional_routing import member_settings, member_settings_scope, activation_receipt
        # Snapshot metadata before admission threads: ContextVars need not be
        # inherited by the worker on every supported Python runtime.
        hydro_settings = {member: member_settings(prepared)
                          for member, prepared in member_inputs_by_id.items()}
        # One atmosphere Experiment can serve distinct prepared hydro tables.
        # Keep those variants for admission, while sharing identical metadata.
        variant_members, seen_variants = [], {}
        for member, prepared in member_inputs_by_id.items():
            key = (id(prepared.experiment), getattr(prepared, "boundary_interval_seconds", None))
            settings = hydro_settings[member]
            seen = seen_variants.setdefault(key, [])
            if settings not in seen:
                seen.append(settings)
                variant_members.append((member, prepared))
        variants = tuple(prepared for _, prepared in variant_members)
        nextgen = [settings is not None and settings.get("enabled") is True
                   for settings in hydro_settings.values()]
        if any(nextgen) or self.request.hydro is not None:
            if not all(nextgen):
                raise ValueError("NextGen ensemble products require hydro enabled for every member")
            from woof.ensemble.nextgen_capture import NextGenCollector
            if not isinstance(self.collector, NextGenCollector):
                self.collector = NextGenCollector(self.collector, self.member_order,
                    options=self.request.hydro, array_module=self.array_module)
        if resumed is not None:
            restore = getattr(self.collector, "restore_resume", None)
            if not callable(restore):
                raise ValueError("ensemble continuation needs its durable diagnostic collector, including completed members")
            restore()
        member_archive = getattr(self.collector, "member_archive", None)
        if self.request.recipe == "wofs":
            from woof.ensemble.wofs_recipe import prewarm_member_kernels
            prewarm_member_kernels(prepared.experiment for prepared in member_inputs_by_id.values())
        if member_archive is not None:
            for member, prepared in member_inputs_by_id.items():
                member_archive.expect_member_forecast(member, prepared.experiment)
        cards, model, scope = self.cards, self.memory_model, self.device_scope
        cp = self.array_module
        profiles, ordinary_admissions, ordinary_sampling = {}, {}, []
        fft_workspaces = {}
        if cards is None or model is None or scope is None:
            if cp is None:
                import cupy as cp
            from woof.core.preflight import estimate_experiment, local_memory_profile_from_device
            from woof.ensemble.prepared_execution import member_device_ids_for_request
            from woof.ensemble.production_memory import ordinary_ensemble_memory_model
            scope = scope or (lambda device: cp.cuda.Device(device))
            if cards is None:
                ids = member_device_ids_for_request(inputs, self.request,
                    visible_count=int(cp.cuda.runtime.getDeviceCount()),
                    current_device=int(cp.cuda.runtime.getDevice()))
                readings = []
                for device in ids:
                    with scope(device):
                        if self.stochastic_provider is not None:
                            fft_workspaces[device] = self.stochastic_provider.sample_fft_workspace_bytes(
                                exp, array_module=cp,
                                other_experiments=tuple(prepared.experiment for prepared in variants
                                                        if prepared.experiment is not exp))
                        from woof.core.preflight import release_unreachable_device_memory
                        release_unreachable_device_memory(cp)
                        free, total = cp.cuda.runtime.memGetInfo()
                        reusable = int(cp.get_default_memory_pool().free_bytes())
                        props = cp.cuda.runtime.getDeviceProperties(device)
                        name = props["name"]
                        if isinstance(name, bytes):
                            name = name.decode()
                        readings.append(CardBudget(device, min(int(total), int(free) + reusable), int(total), str(name)))
                        ordinary_sampling.append({"device_id": device, "driver_free_bytes": int(free),
                            "own_pool_reusable_bytes": reusable, "sample_timing": "before_member_initialization",
                            "stochastic_fft_workspace_bytes": fft_workspaces.get(device, 0)})
                cards = tuple(readings)
            if model is None:
                model = {}
                for card in cards:
                    with scope(card.device_id):
                        profile = local_memory_profile_from_device(cp, device_id=card.device_id)
                        profiles[card.device_id] = profile
                        if card.device_id not in fft_workspaces and self.stochastic_provider is not None:
                            fft_workspaces[card.device_id] = self.stochastic_provider.sample_fft_workspace_bytes(exp,
                                array_module=cp, other_experiments=tuple(prepared.experiment for prepared in variants
                                                                        if prepared.experiment is not exp))
                        from woof.core.mynn_pbl_scratch import mynn_pricing_memory
                        with mynn_pricing_memory(total_bytes=card.total_bytes, free_bytes=card.available_bytes):
                            estimates = []
                            for member, prepared in variant_members:
                                with member_settings_scope(prepared, inherited_settings=hydro_settings[member]):
                                    estimates.append((estimate_experiment(prepared.experiment,
                                        forcing_interval_seconds=prepared.boundary_interval_seconds,
                                        profile=profile, vram_gib=card.total_bytes / (1024 ** 3)), prepared.experiment))
                        estimate, forecast_exp = estimates[0]
                        priced, external, margin = ordinary_ensemble_memory_model(estimate, forecast_exp, self.collector,
                            inventory_id=f"ordinary-ensemble-device-{card.device_id}",
                            stochastic_provider=self.stochastic_provider,
                            fft_workspace_bytes=fft_workspaces.get(card.device_id, 0), other_variants=estimates[1:],
                            initial_wind=is_initial_perturbation(self.request.perturbation))
                        model[card.device_id] = priced
                        ordinary_admissions[card.device_id] = (external, margin)
        cards = tuple(cards)
        if (checkpoint_members and len(cards) > 1
                and not callable(getattr(self.collector, "checkpoint_hold", None))):
            # A roster seal hashes the files the collector snapshot names.
            # Members on other cards could retire those pending packs between
            # the snapshot and the hash unless the collector holds their
            # submissions (restart_roster.checkpoint_hold). The production
            # collector does; a collector without that hold stays on one card.
            import sys
            print(f"ensemble: checkpointed members use one card (device {cards[0].device_id}) of "
                  f"{len(cards)}: this diagnostic collector has no checkpoint_hold, so its snapshot "
                  "could name packs another card retires. Set restart_interval_s = 0 to use every card, "
                  "or use the default ensemble diagnostic collector.", file=sys.stderr, flush=True)
            cards = cards[:1]
            if isinstance(model, Mapping):
                model = {card.device_id: model[card.device_id] for card in cards}
        resolved_request = replace(self.request, member_device_ids=tuple(card.device_id for card in cards))
        from woof.ensemble.progress import (
            EnsembleProgressAdapter, MemberTerminalProgress, progress_host)
        parameters = inspect.signature(runner).parameters
        progress_key = ("observer" if "observer" in runner_options or
                        ("observer" in parameters and "progress_callback" not in parameters)
                        else "progress_callback")
        # One stop request for the whole run. Members observe it at each
        # step through this adapter; the waves own raising and joining.
        run_control = MemberRunControl()
        host = progress_host(runner_options.get(progress_key))
        progress_adapter = EnsembleProgressAdapter(host,
            member_ids=self.member_order, run_seconds=float(exp.run_seconds),
            control=run_control,
            check_products=getattr(self.collector, "check_products", None),
            # A run with no progress host of its own (woof go, woof
            # ensemble, woof sim) still says where each member is.
            terminal=(None if host is not None else
                      MemberTerminalProgress(self.member_order, float(exp.run_seconds))))
        if resumed is not None:
            prior = json.loads((out / "ensemble-run.json").read_text(encoding="utf-8"))
            prior_progress = {row["member_id"]: row for row in
                prior.get("ensemble_progress", {}).get("members", ())}
            from woof.ensemble.restart_roster import resolve_file
            for member, row in resume_rows.items():
                progress_adapter.restore_member(member, elapsed_seconds=row["model_elapsed_seconds"],
                    outer_step=(prior_progress.get(member, {}).get("outer_step", 0)
                        if row["status"] == "completed" else 0),
                    checkpoint=(resolve_file(out, row["checkpoint"]) if row["status"] == "checkpoint" else None))
        has_radar = any(radar_enabled(prepared) for prepared in member_inputs_by_id.values())
        native = (self.native_executor is not None and self.source_execution is None
                  and not has_radar and not checkpoint_members and not any(nextgen))
        original_reasons = ({member: ["NextGen retains its original per-member resident driver"]
                             for member in member_inputs_by_id} if any(nextgen) else {})
        # These selections already require the original model. Diagnose them
        # before starting a first-member native probe so all ordinary members
        # can enter the admitted concurrent wave together.
        from woof.ensemble.ordinary_execution import OrdinaryRecipeExecution
        original_source = self.source_execution is None or isinstance(self.source_execution, OrdinaryRecipeExecution)
        if getattr(self.collector, 'cyclone_request', None) is not None:
            native = False
            original_reasons = {member: ['every-step cyclone swaths retain each original member callback']
                                for member in member_inputs_by_id}
        if self.request.members > 1 and self.native_executor is None and original_source:
            from woof.ensemble.suite_capabilities import plan_suite
            for member, prepared in member_inputs_by_id.items():
                domains = tuple(getattr(prepared.experiment, "domains", ()))
                reasons = []
                for domain in domains:
                    reasons.extend(f"domain {domain.grid_id}: {reason}" for reason in
                                   plan_suite(domain.run, members=self.request.members).native_fallback_reasons)
                if len(domains) > 1:
                    reasons.append("nested members retain the original parent FORCE, child clocks and feedback")
                if reasons:
                    original_reasons[member] = list(dict.fromkeys(original_reasons.get(member, []) + reasons))
        ordinary_cuda_available = cp is not None and callable(getattr(getattr(cp, "cuda", None), "Stream", None))
        from woof.ensemble.ordinary_concurrency import plan_ordinary_concurrency
        concurrency_plans = {member: plan_ordinary_concurrency(prepared.experiment)
            for member, prepared in member_inputs_by_id.items()}
        for member, concurrency in concurrency_plans.items():
            if not concurrency.eligible and member in original_reasons:
                original_reasons[member].extend(concurrency.reasons)
        checkpoint_concurrent = (not checkpoint_members or
            callable(getattr(self.collector, "checkpoint_hold", None)))
        concurrent_ordinary = bool(checkpoint_concurrent and
            (original_reasons or checkpoint_members or self.request.recipe == "wofs") and ordinary_cuda_available
            and not has_radar and original_source and all(
                int(getattr(getattr(prepared.experiment, "devices", None), "count", 1)) <= 1
                for prepared in member_inputs_by_id.values())
            and all(concurrency.eligible for concurrency in concurrency_plans.values()))
        if concurrent_ordinary:
            # Private member allocators cannot borrow cached default-pool
            # blocks. Return unused blocks before charging the sampled budget
            # that included them. Live immutable tables are unaffected.
            for card in cards:
                with scope(card.device_id):
                    cp.cuda.get_current_stream().synchronize()
                    cp.get_default_memory_pool().free_all_blocks()
        from woof.ensemble.packing_policy import ordinary_member_policy
        ordinary_cap, cap_policy = ordinary_member_policy(self.request, concurrent=concurrent_ordinary)
        packing = pack_members(self.request.members, tuple(cards), model, batched=native,
            concurrent_ordinary=concurrent_ordinary,
            max_ordinary_members_per_device=ordinary_cap,
            reason=("qualified native member executor" if native else
                    "posted member sources retain original native initialization and streaming seals"
                    if self.source_execution is not None else
                    "simulated radar retains original member volume writers and live native consumers"
                    if has_radar else
                    "ordinary runner preserves this configuration and each member clock"))
        component_plans, component_rows = {}, []
        component_candidate = (not checkpoint_members and self.request.members > 1 and self.native_executor is None and
            original_source and not has_radar and ordinary_cuda_available and concurrent_ordinary)
        if component_candidate:
            from woof.ensemble.production_component_route import source_component_route_reason
            accepts_handoff = ("schedule_dispatch" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()))
            source_reason = source_component_route_reason(self.source_execution)
            for batch in packing.batches:
                if batch.execution_mode != "ordinary_concurrent_members" or batch.members < 2:
                    continue
                ids = tuple(self.member_order[slot] for slot in batch.member_indices)
                reason = (source_reason or (None if accepts_handoff else
                    "the original member runner has no schedule_dispatch handoff for its writer and controller owners"))
                decision = None
                if reason is None:
                    card = next(card for card in cards if card.device_id == batch.device_id)
                    with scope(card.device_id):
                        decision, reason = _prepared_component_route_plan(
                            {member: member_inputs_by_id[member] for member in ids}, card=card,
                            collector=self.collector, array_module=cp,
                            device_profile=profiles.get(card.device_id), stochastic_provider=self.stochastic_provider)
                key = (batch.wave, batch.device_id)
                if decision is not None:
                    component_plans[key] = decision.reservation
                    for member in ids:
                        original_reasons.pop(member, None)
                else:
                    for member in ids:
                        original_reasons[member] = [reason]
                component_rows.append({"wave": batch.wave, "device_id": batch.device_id,
                    "member_ids": list(ids), "eligible": decision is not None,
                    "ordinary_reason": reason, "complete_native_rk_graph": False,
                    **({} if decision is None else {"memory": decision.reservation.receipt()})})
        from woof.ensemble.lineage import lineage_summary
        manifest = {"lineage": lineage_summary(self.request.provenance), "schema": "gpuwm-ensemble-run.v1", "status": "running",
                    "hydro_activation": [dict(member_id=member, **activation_receipt(hydro_settings[member]))
                                         for member in self.member_order],
                    "request": self.request.receipt(), "packing": packing.receipt(),
                    "ordinary_member_cap_policy": cap_policy,
                    **({} if not self.request.member_variants else {"member_variants": [
                        self._variant_receipt(member) for member in self.member_order]}),
                    # Stated only by an engine gate that runs copies on purpose.
                    **({} if self.identical_members is None else
                       {"identical_members": self.identical_members}),
                    "members_completed": [member for member, row in resume_rows.items() if row["status"] == "completed"],
                    "completed_member_results": [{"member_id": member, "result": row["result"]}
                        for member, row in resume_rows.items() if row["status"] == "completed"],
                    "requested_run_seconds": float(exp.run_seconds),
                    "resumed_members": [{"member_id": member, "status": row["status"],
                        "model_elapsed_seconds": row["model_elapsed_seconds"]} for member, row in resume_rows.items()],
                    "restart_policy": ("original per-member runners and durable member restart roster" if checkpoint_members else "unchanged"),
                    "member_history_files": [],
                    "member_radar_products": [],
                    "member_order": list(self.member_order),
                    "source_roster": (None if self.member_roster is None else self.member_roster.receipt()),
                    **({"posted_source_execution": self.source_execution.receipt()}
                       if self.source_execution is not None else {}),
                    "runtime_stochastic_authorities": [dict(self._stochastic_authorities[id(prepared)], member_id=member)
                        for member, prepared in member_inputs_by_id.items() if id(prepared) in self._stochastic_authorities],
                    "ordinary_memory_sampling": ordinary_sampling,
                    "ordinary_fallback_reasons": [dict(member_id=member, reasons=reasons)
                        for member, reasons in original_reasons.items()],
                    "ordinary_concurrency": [dict(member_id=member, **plan.receipt())
                        for member, plan in concurrency_plans.items()],
                    "ordinary_memory_inventory": [
                        (model[card.device_id] if isinstance(model, Mapping) else model).inventory(1)
                        for card in cards],
                    "shared_preparation": True,
                    "time_step_policy": "ordinary per-member clock; qualified fixed-clock native packs",
                    "streaming_input_policy": "unchanged from prepared input authority"}
        if component_rows:
            manifest["prepared_component_admission"] = component_rows
            manifest["prepared_component_waves"] = []
            if component_plans:
                manifest["time_step_policy"] = "original per-member adaptive clocks; native components join only ready compatible groups"
        if warnings:
            manifest["warnings"] = warnings
        if self._initial_receipts or self.request.perturbation is not None and is_initial_perturbation(self.request.perturbation):
            manifest["initial_perturbations"] = []
        provenance_directories = set()
        for row in self.request.provenance:
            member_directory = out / "members" / f"member-{row['member']:04d}"
            # Claim once before execution. Only a verified restart may reuse an
            # existing directory; ordinary runners below use this same claim.
            member_directory.mkdir(parents=True, exist_ok=self.restart_roster is not None)
            provenance_directories.add(member_directory)
            _atomic_json(member_directory / "member-provenance.json", row)
            if row.get("parent_family"):
                print(f"ensemble lineage: m{row['member']:04d} {row['parent_family']} {row['model_key']} family_weight={row['family_weight']:g}")
        _atomic_json(out / "ensemble-run.json", manifest)
        if original_reasons:
            import sys
            for member, reasons in original_reasons.items():
                print(f"ensemble: member {member} uses its ordinary runner: " + "; ".join(reasons),
                      file=sys.stderr, flush=True)
        # Probability publication belongs to the roster collector. This lock
        # protects durable run progress when independent cards complete together.
        import threading
        manifest_lock = threading.Lock()
        bootstrap_owners, bootstrap_receipts = [], []

        def ordinary_inputs(member_inputs, batch, *, member):
            """Price the tile door in the same hydro scope as its forecast."""
            with member_settings_scope(member_inputs, inherited_settings=hydro_settings[member]):
                from woof.ensemble.prepared_execution import ordinary_member_execution_inputs
                external_margin = ordinary_admissions.get(batch.device_id)
                if external_margin is None:
                    if batch.execution_mode != "ordinary_streamed_member":
                        return member_inputs, None
                    supplied = model[batch.device_id] if isinstance(model, Mapping) else model
                    external = tuple(component for component in supplied.components
                                     if component.category in ("products", "health", "stochastic"))
                    if external:
                        external_margin = (external, supplied.allocator_margin)
                if external_margin is None:
                    from woof.ensemble.production_memory import ordinary_output_memory_components
                    from woof.ensemble.admission import AllocatorMargin
                    from woof.core.preflight import FORECAST_POOL_HEADROOM
                    external_margin = (ordinary_output_memory_components(member_inputs.experiment, self.collector),
                        AllocatorMargin.from_fraction(str(FORECAST_POOL_HEADROOM - 1),
                            evidence="original pool headroom for additional ensemble output reservations"))
                if (batch.execution_mode == "ordinary_streamed_member" and self.stochastic_provider is not None
                        and self.stochastic_provider.enabled_for_experiment(member_inputs.experiment)):
                    from woof.ensemble.production_memory import ordinary_stochastic_streamed_inputs
                    return ordinary_stochastic_streamed_inputs(member_inputs, batch=batch,
                        external_components=external_margin[0], allocator_margin=external_margin[1],
                        stochastic_provider=self.stochastic_provider,
                        fft_workspace_bytes=fft_workspaces.get(batch.device_id, 0),
                        device_profile=profiles.get(batch.device_id))
                return ordinary_member_execution_inputs(member_inputs, batch=batch,
                    external_components=external_margin[0], allocator_margin=external_margin[1])

        # The automatic factory's first original initializer also needs the
        # cold zero-fit overlay. This wrapper is reused for its ordinary tails.
        def admitted_runner(member_inputs, *, output_directory, **options):
            from woof.ensemble.runtime_context import current_capture
            capture = current_capture()
            member = self.member_order[0] if capture is None else capture.member_id
            device = cards[0].device_id if cp is None else int(cp.cuda.runtime.getDevice())
            batches = [batch for batch in packing.batches if batch.device_id == device
                       and self.member_order.index(member) in batch.member_indices]
            if not batches:
                batches = [batch for batch in packing.batches if batch.device_id == device]
            if batches:
                batch = batches[0]
            else:
                card = next(card for card in cards if card.device_id == device)
                card_model = model[device] if isinstance(model, Mapping) else model
                need = card_model.required_bytes(1)
                mode = "ordinary_member" if need <= card.available_bytes else "ordinary_streamed_member"
                batch = MemberBatch(0, device, (0,), mode,
                    need if mode == "ordinary_member" else None, card.available_bytes,
                    "cold admission for an unused original per-card bootstrap")
            member_inputs, overlay = ordinary_inputs(member_inputs, batch, member=member)
            options[progress_key] = (None if capture is None else progress_adapter.callback_for_member(member))
            result = runner(member_inputs, output_directory=output_directory, **options)
            if overlay is not None and overlay.get("changed"):
                result = asdict(result) if is_dataclass(result) else dict(result)
                result["ensemble_execution_overlay"] = overlay
            return result

        def original_card_bootstrap(*, device_id, shared_inputs, request):
            """Restore the original prepared source on its own card, no step."""
            from woof.ensemble.prepared_execution import InitializedCardBootstrap, sample_initialized_card_evidence
            from woof.ensemble.runtime_context import member_output_scope
            holder = {}
            path = out / ".ensemble-initialization" / f"card-{device_id:02d}"
            path.mkdir(parents=True, exist_ok=False)
            def capture(**handoff):
                from woof.ensemble.production_memory import require_bootstrap_device
                require_bootstrap_device(handoff["node"], array_module=cp, device_id=device_id)
                sample = sample_initialized_card_evidence(handoff["inputs"], array_module=cp)
                if sample.card.device_id != device_id:
                    raise RuntimeError("original bootstrap initialized a different physical card")
                holder.update(handoff, evidence=sample)
                return {"status": "PASS", "ensemble_bootstrap_only": True,
                        "completed_seconds": 0.0, "forecast_steps": 0}
            options = dict(runner_options, first_products=None, ensemble_bootstrap=capture)
            options[progress_key] = None
            with scope(device_id), member_output_scope(None):
                report = admitted_runner(shared_inputs, output_directory=path, **options)
            if (not holder or not isinstance(report, dict) or report.get("status") != "PASS"
                    or report.get("forecast_steps") != 0):
                raise RuntimeError("original per-card bootstrap did not stop before its first forecast step")
            bootstrap_owners.append(holder)
            bootstrap_receipts.append({"device_id": device_id, "path": str(path.relative_to(out)),
                "forecast_steps": 0, "evidence": holder["evidence"].receipt(),
                "source": "same original prepared input; fresh original model and clock on this card"})
            return InitializedCardBootstrap(holder["inputs"], holder["node"], holder["evidence"])

        automatic = None
        if (not any(nextgen) and not checkpoint_members and not has_radar and self.source_execution is None and self.native_executor is None and hasattr(inputs, "execution_plan")
                and self.cards is None and self.memory_model is None
                and not component_plans
                and not (original_reasons and ordinary_cuda_available)):
            from woof.ensemble.prepared_execution import make_automatic_prepared_executor
            automatic = make_automatic_prepared_executor(admitted_runner, request=resolved_request,
                collector=self.collector, output_directory=out, member_roster=self.member_roster,
                input_provider=(None if self.input_provider is None else
                    lambda *, member_id, **kwargs: member_inputs_by_id[member_id]), device_scope=scope,
                initialize_callback_factory=self._initialization_callback,
                card_bootstrap_factory=self.card_bootstrap_factory or original_card_bootstrap,
                array_module=cp,
                stochastic_enabled=(self.stochastic_provider is not None and
                    self.stochastic_provider.enabled_for_experiment(exp)))
        first_result = None

        def execute_component_wave(batch, reservation):
            # Return only finalized records before collecting joined model cycles.
            from woof.ensemble.prepared_nested_batch import ProductionNestedPackFactory
            from woof.ensemble.production_component_route import run_production_component_wave
            def source_context(*, member_id, actual_inputs, authority, source_receipt):
                with manifest_lock:
                    if authority is not None:
                        rows = {row["member_id"]: row for row in manifest["runtime_stochastic_authorities"]}
                        rows[member_id] = dict(authority, member_id=member_id)
                        manifest["runtime_stochastic_authorities"] = [rows[member]
                            for member in self.member_order if member in rows]
                    manifest["posted_source_execution"] = source_receipt
                    _atomic_json(out / "ensemble-run.json", manifest)
            factory = ProductionNestedPackFactory(qualification_receipt={
                "scope": "original prepared runners with qualified parent/nest and production physics components",
                "default_route_enabled": True, "complete_native_rk_graph": False},
                reuse_original_workspaces=_COMPONENT_WORKSPACE_REUSE_QUALIFIED,
                edge_state_only=_COMPONENT_EDGE_STATE_ONLY_QUALIFIED)
            with scope(batch.device_id):
                joined = run_production_component_wave(self, runner,
                    member_inputs={member: member_inputs_by_id[member] for member in batch.member_indices},
                    output_directory=out, reservation=reservation,
                    progress_adapter=progress_adapter, control=run_control,
                    runner_options=runner_options, progress_key=progress_key, device_id=batch.device_id,
                    operation_authority=lambda *, member_id, **unused: {
                        "prepared_authority_sha256": dict(getattr(member_inputs_by_id[member_id], "authority_sha256", {}) or {})},
                    source_context_callback=(source_context if self.source_execution is not None else None),
                    array_module=cp, pack_factory=factory)
            results = []
            for member in batch.member_indices:
                single = replace(batch, member_indices=(member,), execution_mode="ordinary_member",
                    required_bytes=batch.required_bytes // batch.members)
                record = finalize_member_result(single, joined.member_results[member],
                    member_inputs=member_inputs_by_id[member], capture=joined.captures[member],
                    member_out=joined.output_directories[member])
                results.append({"member_id": member, "result": record})
            route = _component_route_receipt(joined.execution)
            execution = route["execution"]
            packed = any(row.get("packed") for row in execution.get("operations", ()))
            with manifest_lock:
                manifest["prepared_component_waves"].append({"wave": batch.wave,
                    "device_id": batch.device_id, "member_ids": list(batch.member_indices),
                    "native_component_operations": packed, "complete_native_rk_graph": False,
                    "route": route})
                manifest["prepared_component_waves"].sort(key=lambda row: (row["wave"], row["device_id"]))
                _atomic_json(out / "ensemble-run.json", manifest)
            return {"status": "PASS", "backend": ("original_members_with_native_components"
                if packed else "ordinary_concurrent_members"), "members": results,
                "complete_native_rk_graph": False, "component_route": route}

        def execute(batch):
            # Packing owns local slots. All source, RNG and output interfaces
            # retain the original global member identifiers.
            batch = replace(batch, member_indices=tuple(self.member_order[slot] for slot in batch.member_indices))
            try:
                if batch.execution_mode == "ordinary_concurrent_members":
                    reservation = component_plans.get((batch.wave, batch.device_id))
                    if reservation is not None:
                        with scope(batch.device_id), finished_member_release(cp):
                            return execute_component_wave(batch, reservation)
                    from woof.ensemble.member_stream import member_cuda_scope
                    return execute_concurrent_members(batch, execute_admitted,
                        member_scope=lambda **kwargs: member_cuda_scope(array_module=cp, **kwargs),
                        control=run_control)
                if concurrent_ordinary and batch.execution_mode == "ordinary_member":
                    # A one-member wave on each of several cards still needs
                    # independent grid-ID CFL banks and physics cache owners.
                    from woof.ensemble.member_stream import member_cuda_scope
                    with member_cuda_scope(array_module=cp, device_id=batch.device_id,
                                           member_id=batch.member_indices[0]) as owned:
                        result = execute_admitted(batch)
                    return dict(result, cuda_scope=owned.receipt())
                if (batch.execution_mode != "member_batched" and int(getattr(getattr(
                        member_inputs_by_id[batch.member_indices[0]].experiment, "devices", None), "count", 1)) <= 1):
                    # Sequential models retain the original stream and pool.
                    # Independent cards still cannot share grid-ID CFL banks.
                    from woof.core.cfl_member import member_cfl_scope
                    with member_cfl_scope():
                        try:
                            return execute_admitted(batch)
                        finally:
                            current_stream = getattr(getattr(cp, "cuda", None), "get_current_stream", None)
                            if current_stream is not None:
                                with scope(batch.device_id):
                                    current_stream().synchronize()
                return execute_admitted(batch)
            except Exception as error:
                # The error says which member it belongs to, in its own
                # sentence and in the manifest, by the original member IDs.
                name_failing_members(error, batch.member_indices, device_id=batch.device_id,
                    wave=batch.wave, execution_mode=batch.execution_mode)
                raise

        def finalize_member_result(batch, result, *, member_inputs=None, capture=None, member_out=None, overlay=None):
            """Publish the same original report/capture after either joined route."""
            member = batch.member_indices[0]
            record = asdict(result) if is_dataclass(result) else result
            if self.request.member_variants:
                if batch.execution_mode == "member_batched":
                    record = dict(record, member_variants=[self._variant_receipt(index)
                        for index in batch.member_indices])
                else:
                    record = dict(record, member_variant=self._variant_receipt(member))
            if batch.execution_mode != "member_batched" and member in self._surface_receipts:
                record = dict(record, surface_perturbation=self._surface_receipts[member])
            if batch.execution_mode != "member_batched" and member in self._initial_receipts:
                record = dict(record, initial_perturbation=self._initial_receipts[member])
            branch = getattr(self, "_checkpoint_fill", None)
            if branch is not None and member in branch.receipts:
                record = dict(record, checkpoint_branches=list(branch.receipts[member].values()))
                if branch.reconstructions.get(member):
                    record = dict(record, checkpoint_reconstructions=list(branch.reconstructions[member]))
            if batch.execution_mode != "member_batched" and overlay is not None and overlay.get("changed"):
                record = dict(record, ensemble_execution_overlay=overlay)
            if batch.execution_mode != "member_batched" and capture.counter_calendars:
                record = dict(record)
                record["ensemble_counter_observations"] = [
                    calendar.receipt(member_id=member, episode=episode)
                    for calendar in capture.counter_calendars.values()
                    for observed_member, episode in calendar._progress
                    if observed_member == member]
            if isinstance(record, dict) and record.get("status") not in (None, "PASS"):
                raise RuntimeError("ordinary member returned a failing forecast receipt")
            radar_products = (None if batch.execution_mode == "member_batched" else
                finish_member_radar(member_inputs, member_out, result,
                                    keep_member_files=self.request.keep_member_files))
            with manifest_lock:
                if radar_products is not None:
                    manifest["member_radar_products"].append(dict(radar_products,
                        member_id=member, directory=str(member_out.relative_to(out))))
                    manifest["member_radar_products"].sort(key=lambda row: self.member_order.index(row["member_id"]))
                manifest["members_completed"].extend(batch.member_indices)
                manifest["members_completed"].sort()
                if batch.execution_mode != "member_batched":
                    manifest["completed_member_results"].append({"member_id": member, "result": record})
                manifest["surface_perturbations"] = [self._surface_receipts[member]
                    for member in self.member_order if member in self._surface_receipts]
                if self._initial_receipts:
                    manifest["initial_perturbations"] = [self._initial_receipts[member]
                        for member in self.member_order if member in self._initial_receipts]
                if self.source_execution is not None:
                    manifest["posted_source_execution"] = self.source_execution.receipt()
                _atomic_json(out / "ensemble-run.json", manifest)
                if checkpoint_members:
                    from woof.ensemble.restart_roster import seal
                    save = getattr(self.collector, "save_resume", None)
                    try:
                        seal(out, collector=self.collector if callable(save) else None)
                    except TimeoutError:
                        # The product drain is still busy past its bounded
                        # hold. As in CheckpointProgress, the snapshot is
                        # deferred and the previous roster stays the durable
                        # one: the end-of-run path drains with no bound
                        # (require_complete) and then seals the completed
                        # tree. An unbounded wait here would hold the
                        # collector lock and stop every other member's frame
                        # submissions for the whole backlog. A finished
                        # member must not fail the run on this
                        # (run_4a75cc2fe19d92f7, 6 h HRRR WoFS with a 1 km
                        # nest, 2026-10-28).
                        reason = "product drain exceeded bounded checkpoint hold"
                        manifest.setdefault("ensemble_checkpoint_deferrals", []).append(
                            {"member_ids": list(batch.member_indices), "at": "member_completion",
                             "reason": reason})
                        if isinstance(record, dict):
                            record = dict(record, ensemble_checkpoint_deferred=reason)
                            if batch.execution_mode != "member_batched":
                                # Appended above under this same lock.
                                manifest["completed_member_results"][-1]["result"] = record
                        _atomic_json(out / "ensemble-run.json", manifest)
            return record

        def execute_admitted(batch):
            member_inputs = capture = member_out = overlay = None
            with scope(batch.device_id):
                if batch.execution_mode == "member_batched":
                    native_options = dict(runner_options)
                    native_options[progress_key] = progress_adapter.callback_for_member(batch.member_indices[0])
                    result = self.native_executor(inputs=inputs, batch=batch,
                        request=self.request, collector=self.collector,
                        output_directory=out, input_provider=self.input_provider,
                        runner_options=native_options)
                    if result is None:
                        raise RuntimeError("admitted native pack returned no forecast receipt")
                else:
                    member = batch.member_indices[0]
                    resume = resume_rows.get(member, {})
                    if resume.get("status") == "completed":
                        return resume["result"]
                    member_inputs = member_inputs_by_id[member]
                    member_inputs, overlay = ordinary_inputs(member_inputs, batch, member=member)
                    initialize = self._initialization_callback(member)
                    if any(nextgen):
                        previous_initialize = initialize
                        def initialize(**kwargs):
                            self.collector.bind_member(member_id=member, **kwargs)
                            if previous_initialize is not None:
                                previous_initialize(**kwargs)
                    retained_models = []
                    if concurrent_ordinary:
                        previous = initialize
                        def initialize(**kwargs):
                            # Hold construction owners while this worker's
                            # queue is active. Another member's GC pass cannot
                            # collect a completed asynchronous model early.
                            retained_models.append(kwargs.get("model") or kwargs.get("state"))
                            if previous is not None:
                                previous(**kwargs)
                    capture = MemberOutputCapture(self.collector.submit, member,
                        member_history_required(member_inputs, self.request.keep_member_files),
                        initialize_callback=initialize,
                        provenance=(self.request.provenance[member] if self.request.provenance else None))
                    options = dict(runner_options)
                    options["first_products"] = None
                    member_progress = progress_adapter.callback_for_member(member)
                    if checkpoint_members:
                        from woof.ensemble.restart_roster import CheckpointProgress
                        member_progress = CheckpointProgress(member_progress, collector=self.collector,
                            root=out, manifest_lock=manifest_lock)
                    options[progress_key] = member_progress
                    member_out = out / "members" / f"member-{member:04d}"
                    if resume.get("status") == "checkpoint":
                        from woof.ensemble.restart_roster import resolve_file
                        options["restart"] = resolve_file(out, resume["checkpoint"])
                    if first_result is not None and member == first_result.first_member_id:
                        result = first_result.report
                    else:
                        if member_out not in provenance_directories:
                            member_out.mkdir(parents=True, exist_ok=self.restart_roster is not None)
                        # Every original member retains its own model until
                        # its queue completes, including concurrent waves.
                        from woof.ensemble.timeline import member_timeline
                        with finished_member_release(cp), member_output_scope(capture), member_timeline(
                                member_out, member=member, card=batch.device_id):
                            if self.source_execution is None:
                                with member_settings_scope(member_inputs, inherited_settings=hydro_settings[member]):
                                    result = runner(member_inputs, output_directory=member_out, **options)
                            else:
                                def forecast(actual_inputs):
                                    nonlocal overlay
                                    actual_inputs = self._configured_member_inputs(actual_inputs)
                                    authority = self._stochastic_authorities.get(id(actual_inputs))
                                    actual_inputs, overlay = ordinary_inputs(actual_inputs, batch, member=member)
                                    with manifest_lock:
                                        if authority is not None:
                                            rows = {row["member_id"]: row for row in
                                                manifest["runtime_stochastic_authorities"]}
                                            rows[member] = dict(authority, member_id=member)
                                            manifest["runtime_stochastic_authorities"] = [
                                                rows[index] for index in self.member_order if index in rows]
                                        manifest["posted_source_execution"] = self.source_execution.receipt()
                                        _atomic_json(out / "ensemble-run.json", manifest)
                                    with member_settings_scope(actual_inputs, inherited_settings=hydro_settings[member]):
                                        return runner(actual_inputs, output_directory=member_out, **options)
                                result = self.source_execution.run_member(member, forecast=forecast,
                                    observer=options.get(progress_key))
                    if self.request.keep_member_files:
                        for path in sorted(member_out.glob("**/wrfout_*")):
                            match = re.match(r"wrfout_d([0-9]+)_", path.name)
                            if match is None or not path.is_file() or path.suffix == ".json":
                                continue
                            episodes = [part for part in path.parts if part.startswith("episode-")]
                            episode = int(episodes[-1].split("-", 1)[1]) if episodes else 0
                            self.collector.add_member_file(path, member_id=member,
                                grid_id=int(match.group(1)), episode=episode)
                return finalize_member_result(batch, result, member_inputs=member_inputs,
                    capture=capture, member_out=member_out, overlay=overlay)

        def publish_member_provenance():
            # Refresh the directories claimed above after execution stops,
            # including pending members when a forecast fails or is interrupted.
            # Never recreate or silently accept another directory here.
            for row in self.request.provenance:
                directory = out / "members" / f"member-{row['member']:04d}"
                _atomic_json(directory / "member-provenance.json", row)

        def surface_inventory():
            from woof.ensemble.surface_recipe import surface_realization_inventory
            receipts = [self._surface_receipts[member] for member in self.member_order
                        if member in self._surface_receipts]
            return surface_realization_inventory(receipts)

        try:
            if automatic is not None:
                try:
                    # A native pack inside this call runs its own waves; they
                    # find the run's stop request through this scope.
                    with member_run_scope(run_control):
                        first_result = automatic(inputs, runner_options=runner_options)
                except Exception as error:
                    # The first member runs here on the calling thread. A
                    # native pack that failed inside it already named itself.
                    name_failing_members(error, (self.member_order[0],))
                    raise
                finally:
                    # Retained original card owners cover every native wave.
                    # Drop them before any fallback tail is initialized.
                    bootstrap_owners.clear()
                    import gc
                    gc.collect()
                manifest["automatic_admission"] = first_result.admission
                manifest["additional_card_bootstraps"] = bootstrap_receipts
            if first_result is not None and first_result.native_complete:
                manifest["members_completed"] = list(self.member_order)
                manifest["packing"] = first_result.report["admission"]["packing"]
                results = ()
                completed_records = first_result.report["member_results"]
            else:
                results = execute_member_packing(packing, execute, control=run_control)
                completed_records = [{"packing": batch.receipt(), "result": result}
                                     for batch, result in results]
            publish_member_provenance()
            products_started = time.monotonic()
            products = self.collector.finish_run()
            self.collector.require_complete()
            if self.source_execution is not None:
                manifest["posted_source_execution"] = self.source_execution.require_complete()
            files = sorted(out.glob("members/**/wrfout_*"))
            files = [path for path in files if path.is_file() and path.suffix != ".json"]
            ledger = getattr(self.collector, "history_ledger", None)
            identities = [] if ledger is None else ledger.validate_retirements()
            histories = ([row["path"] for row in identities] if identities else
                         [str(path.relative_to(out)) for path in files])
            if files and not self.request.keep_member_files:
                raise RuntimeError("aggregate-only execution unexpectedly wrote member histories")
            manifest.update(status="PASS", wall_seconds=time.perf_counter() - started,
                            completed_seconds=float(exp.run_seconds), products=products,
                            ensemble_progress=progress_adapter.receipt(),
                            member_history_files=histories,
                            member_results=completed_records)
            manifest["surface_perturbations"] = [self._surface_receipts[member]
                for member in self.member_order if member in self._surface_receipts]
            manifest["surface_realization_inventory"] = surface_inventory()
            if any(nextgen):
                manifest["nextgen"] = self.collector.nextgen_points.receipt()
                manifest["nextgen"]["resident_inventory"] = self.collector.resident_inventory.receipt()
            if self._initial_receipts:
                manifest["initial_perturbations"] = [self._initial_receipts[member]
                    for member in self.member_order if member in self._initial_receipts]
            if self.request.sounding_points:
                from woof.soundings import render_run
                manifest["products"]["soundings"] = str(render_run(
                    out, points=self.request.sounding_points, out=out / "render"))
            if self.request.cyclone_products is not None:
                from woof.ensemble.cyclone_products import render_roster
                products["cyclone"] = render_roster(out, self.request.trajectories,
                    self.request.cyclone_products, member_order=self.member_order,
                    history_ledger=self.collector.history_ledger)
            manifest["timing_seconds"] = self._timing_seconds(started_monotonic, products_started)
            _atomic_json(out / "ensemble-run.json", manifest)
            self.last_manifest = manifest
            self.last_output_directory = out
            _atomic_json(out / "report.json", manifest)
            _atomic_json(out / "progress.json", {"schema": "gpuwm-ensemble-progress.v1",
                "status": "PASS", "model_elapsed_seconds": manifest["completed_seconds"],
                "requested_run_seconds": float(exp.run_seconds),
                "members_completed": list(manifest["members_completed"]),
                "ensemble_manifest": "ensemble-run.json"})
            if checkpoint_members:
                # The completed tree is the durable one. The roster last
                # sealed at the final member's completion named files that
                # finish_run then retired (the worker-only coordinates.nc)
                # and the ensemble-run.json this PASS manifest replaced, so a
                # completed rc-0 run failed its own restart validator
                # (2.9.0 ensemble showcase, restart_roster.verify_roster).
                from woof.ensemble.restart_roster import seal
                save = getattr(self.collector, "save_resume", None)
                seal(out, collector=self.collector if callable(save) else None)
            return manifest
        except BaseException as error:
            publish_member_provenance()
            cancel = getattr(self.collector, "cancel_products", None)
            if cancel is not None:
                cancel()
            if self.source_execution is not None:
                manifest["posted_source_execution"] = self.source_execution.receipt()
            # A stop is not a member failure: the run was interrupted and
            # every member that had not finished is simply incomplete.
            stopped = isinstance(error, (KeyboardInterrupt, SystemExit))
            completed = set(manifest["members_completed"])
            manifest.update(status="interrupted" if stopped else "failed",
                            wall_seconds=time.perf_counter() - started,
                            ensemble_progress=progress_adapter.receipt(),
                            error_type=type(error).__name__, error=str(error),
                            failed_members=[] if stopped else failed_member_rows(error),
                            members_not_completed=[member for member in self.member_order
                                                   if member not in completed])
            manifest["surface_perturbations"] = [self._surface_receipts[member]
                for member in self.member_order if member in self._surface_receipts]
            if self._initial_receipts:
                manifest["initial_perturbations"] = [self._initial_receipts[member]
                    for member in self.member_order if member in self._initial_receipts]
            manifest["surface_realization_inventory"] = surface_inventory()
            _atomic_json(out / "ensemble-run.json", manifest)
            if checkpoint_members:
                from woof.ensemble.restart_roster import seal
                save = getattr(self.collector, "save_resume", None)
                if callable(save):
                    # The roster seal waits for the product worker this
                    # handler just cancelled, so it can raise that
                    # cancellation ("product process exited -15").  It must
                    # not replace the error that stopped the run: a user saw
                    # only the -15 while the member's own error sat unread
                    # in ensemble-run.json (fresh-install QA, 2026-10-12).
                    try:
                        seal(out, collector=self.collector)
                    except Exception as seal_error:
                        note = (f"the restart roster could not be sealed after "
                                f"this failure: {type(seal_error).__name__}: "
                                f"{seal_error}")
                        manifest["restart_roster_seal_error"] = note
                        _atomic_json(out / "ensemble-run.json", manifest)
                        error.add_note(note)
            raise

    def _timing_seconds(self, started, products_started):
        """Where this run's wall went, as stage_timing.timing_coverage reads it.

        A door that fetched and prepared the members hands its own clocks
        (``door_stage_clock``), so ``total`` is the door's whole wall and the
        forecast is timed from the door's forecast stage; without one, the
        session's own forecast and products.
        """
        now = time.monotonic()
        clock = getattr(self, "door_stage_clock", None)
        if clock is None:
            return {"forecast": products_started - started,
                    "products": now - products_started, "total": now - started}
        return {"plan": clock["plan"], "fetch": clock["fetch"], "prepare": clock["prepare"],
                "forecast": products_started - clock["forecast_started"],
                "products": now - products_started, "total": now - clock["started"]}

    def completed_products(self):
        """Verify durable aggregate manifests before a door closes rendering."""
        if self.last_manifest is None or self.last_manifest["status"] != "PASS":
            raise RuntimeError("ensemble rendering has no completed forecast roster")
        from woof.ensemble.radar_output import verified_radar_products
        for member in self.last_manifest.get("member_radar_products", ()):
            actual = verified_radar_products(self.last_output_directory / member["directory"])
            if actual["manifest_sha256"] != member["manifest_sha256"]:
                raise RuntimeError("completed member radar manifest changed after publication")
        for domain in self.last_manifest["products"].get("domain_manifests", ()):
            path = self.last_output_directory / domain["manifest"]
            document = json.loads(path.read_text(encoding="utf-8"))
            deferred = bool(getattr(self.collector, "defer_replay", False))
            output_order = getattr(self.collector, "member_order", self.member_order)
            if any(frame["status"] not in ("complete", "unavailable") and not
                   (deferred and frame["status"] == "pending" and
                    frame["members_received"] == sorted(output_order))
                   for frame in document["frames"]):
                raise RuntimeError("ensemble product manifest still contains an incomplete roster")
            for frame in document["frames"]:
                for relative in (*frame.get("products", ()), *frame.get("maps", ())):
                    if not (self.last_output_directory / relative).is_file():
                        raise RuntimeError("ensemble product manifest names a missing completed artifact")
        return self.last_manifest["products"]

    def _initialization_callback(self, member_id, *, seed=None, prepared_member=None):
        from woof.ensemble.surface_recipe import is_surface_recipe
        from woof.ensemble.initial_perturbation import is_initial_perturbation, initial_perturbation_callback
        from woof.ensemble.member_variants import member_surface_options
        surface_options = member_surface_options(self.request, member_id)
        wind = (self.request.member_variants[member_id].get("wind", 0.)
                if self.request.member_variants else 0.)
        initial_options = ({"kind": "uniform-wind", "amplitude": wind} if wind else self.request.perturbation)
        checkpoint_fill = (self.request.recipe == "wofs" and self.request.provenance
                           and self.request.provenance[member_id]["rung"] == "perturbation")
        if (self.stochastic_provider is None and not is_surface_recipe(surface_options)
                and not is_initial_perturbation(initial_options) and not checkpoint_fill):
            return None
        prepared = (prepared_member if prepared_member is not None else None
                    if self.member_roster is None else self.member_roster.select((member_id,))[0])
        if prepared is None and self.source_execution is not None:
            prepared = self.source_execution.stochastic_member_binding(member_id)
        seed = self._member_seed(member_id, seed=seed, prepared_member=prepared)
        def initialize(*, model=None, prepared_case=None, state=None, cfg=None, grid=None, clock=None):
            binding = (self.source_execution.stochastic_member_binding(member_id)
                       if self.source_execution is not None else prepared)
            if self.stochastic_provider is None:
                return
            if model is not None:
                self.stochastic_provider.bind_model(model=model, member_id=member_id,
                    seed=seed, prepared_member=binding)
            else:
                self.stochastic_provider.bind_state(state=state, cfg=cfg, clock=clock,
                    member_id=member_id, seed=seed, prepared_member=binding)
        from woof.ensemble.surface_recipe import surface_initialization_callback
        from types import SimpleNamespace
        selected = SimpleNamespace(perturbation=surface_options)
        def record(receipt):
            if self.request.member_variants:
                receipt = dict(receipt, member_variant=self._variant_receipt(member_id,
                    seed=seed, prepared_member=prepared))
            self._surface_receipts[member_id] = receipt
        callback = surface_initialization_callback(selected, member_id=member_id, seed=seed,
            previous=(None if self.stochastic_provider is None else initialize),
            record=record,
            array_module=self.array_module)
        if is_initial_perturbation(initial_options):
            def record_initial(receipt):
                self._initial_receipts[member_id] = receipt
                if self.last_manifest is not None:
                    self.last_manifest["initial_perturbations"] = [self._initial_receipts[member]
                        for member in self.member_order if member in self._initial_receipts]
            callback = initial_perturbation_callback(initial_options,
                member_id=member_id, seed=seed, previous=callback,
                record=record_initial, array_module=self.array_module)
        if checkpoint_fill:
            callback = self._checkpoint_fill.callback(self.request.provenance[member_id],
                member_id=member_id, previous=callback)
        return callback

    def _member_seed(self, member_id, *, seed=None, prepared_member=None):
        """Keep a supplied/source seed authoritative through member callbacks."""
        if seed is not None:
            return seed
        if self.source_execution is not None:
            return self.source_execution.member_metadata[member_id]["seed"]
        if prepared_member is None and self.member_roster is not None:
            prepared_member = self.member_roster.select((member_id,))[0]
        if prepared_member is not None:
            return prepared_member.seed
        from woof.ensemble.seeds import member_seed
        return member_seed(self.request.base_seed, member_id)

    def _variant_receipt(self, member_id, *, seed=None, prepared_member=None):
        variant = self.request.member_variants[member_id]
        if seed is None and member_id in self._surface_receipts:
            seed = self._surface_receipts[member_id]["seed"]
        return {"member_id": member_id, "seed": self._member_seed(member_id,
                    seed=seed, prepared_member=prepared_member),
                "name": variant["name"], "physics": dict(variant["physics"]),
                "surface": dict(variant["surface"]),
                **({"wind": variant["wind"]} if variant.get("wind") else {}),
                **({} if member_id not in self._member_land_layouts else {
                    "resolved_land": [dict(row) for row in self._member_land_layouts[member_id]]}),
                **({} if member_id not in getattr(self, "_member_physics_selections", {}) else {
                    "resolved_physics": self._member_physics_selections[member_id]})}

    def _remember_member_land_layouts(self, member_inputs):
        """Record each member's loaded domain selectors before initialization."""
        if not self.request.member_variants:
            return
        from woof.config import soil_layer_count
        self._member_land_layouts = {member: [
            {"grid_id": int(domain.grid_id),
             "sf_surface_physics": int(domain.run.sf_surface_physics),
             "num_soil_layers": soil_layer_count(domain.run)}
            for domain in inputs.experiment.domains]
            for member, inputs in member_inputs.items()}
        self._member_physics_selections = {}
        from woof.ensemble.member_variants import LAND_KEYS
        from woof.microphysics_schemes import resolve_mp_physics
        for member, rows in self._member_land_layouts.items():
            requested = self.request.member_variants[member]["physics"]
            selected_rows = []
            for row, domain in zip(rows, member_inputs[member].experiment.domains):
                selected = {"grid_id": row["grid_id"]}
                for key, value in requested.items():
                    actual = row[key] if key in LAND_KEYS else getattr(domain.run, key)
                    expected = resolve_mp_physics(value) if key == "mp_physics" else value
                    if actual != expected:
                        raise ValueError(
                            f"member {member} domain {row['grid_id']} requests {key}={value}, "
                            f"but its input provider bound {key}={actual}; " +
                            ("the wrong land column would run this member's forecast" if key in LAND_KEYS
                             else "the wrong physics selection would run this member's forecast"))
                    selected[key] = actual
                selected_rows.append(selected)
            if set(requested) - LAND_KEYS:
                self._member_physics_selections[member] = selected_rows

    def _configured_member_inputs(self, inputs):
        if self.stochastic_provider is None:
            return inputs
        previous = self._stochastic_input_bindings.get(id(inputs))
        if previous is not None:
            original, bound = previous
            if original is not inputs:
                raise AssertionError("stochastic input owner identity changed inside the active session")
            return bound
        from woof.ensemble.stochastic_authority import configure_member_inputs
        bound, authority = configure_member_inputs(self.stochastic_provider, inputs)
        self._stochastic_input_bindings[id(inputs)] = (inputs, bound)
        self._stochastic_input_bindings[id(bound)] = (bound, bound)
        if authority is not None:
            self._stochastic_authorities[id(bound)] = authority
        return bound

    def run_experiment(self, runner, experiment, case_data, output_directory, **runner_options):
        """Reuse native source preparation around the original runtime door."""
        if self.request.recipe == "wofs" and self.input_provider is None:
            from woof.ensemble.wofs_recipe import runtime_member_provider
            self.input_provider = runtime_member_provider(self.request)
        self._require_member_inputs(experiment)
        from woof.ensemble.runtime_preparation import RuntimeMemberInputs, RuntimePreparationSource
        from woof.runtime import ExperimentRunSummary
        out = Path(output_directory)
        source = RuntimePreparationSource(out / ".ensemble-preparation")
        shared = RuntimeMemberInputs(experiment, case_data)
        def member_runner(member_inputs, *, output_directory, first_products=None, **options):
            if not isinstance(member_inputs, RuntimeMemberInputs):
                raise TypeError("config-driven source members require their native RuntimeMemberInputs binding")
            with source.scope(member_inputs.member_input):
                return runner(member_inputs.experiment, member_inputs.case_data,
                              output_directory, **options)
        try:
            manifest = self.run_prepared(member_runner, shared,
                output_directory=out, **runner_options)
            preparation = source.receipt()
        finally:
            source.close()
        preparation["deleted_files"] = source.receipt()["deleted_files"]
        preparation["closed"] = True
        manifest["shared_native_preparation"] = preparation
        _atomic_json(out / "ensemble-run.json", manifest)
        _atomic_json(out / "report.json", manifest)
        ensemble_manifest = out / "ensemble-run.json"
        return ExperimentRunSummary(
            wrfout_paths=tuple(out / relative for relative in manifest["member_history_files"]),
            completed_seconds=manifest["completed_seconds"], nan_free=True,
            ensemble_manifest=ensemble_manifest,
            ensemble_manifest_sha256=hashlib.sha256(ensemble_manifest.read_bytes()).hexdigest())


def run_prepared_ensemble(inputs, *, runner, request, output_directory,
                          member_roster=None, input_provider=None, stochastic_provider=None,
                          source_execution=None, session_options=None, **options):
    """Run a prepared ensemble through an original prepared forecast door."""
    return PreparedEnsembleSession(request, output_directory=output_directory,
        member_roster=member_roster, input_provider=input_provider,
        stochastic_provider=stochastic_provider, source_execution=source_execution,
        **(session_options or {})).run_prepared(
        runner, inputs, **options)
