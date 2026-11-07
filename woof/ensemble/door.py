"""One ensemble option shared by the ordinary configuration front doors."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import tomllib

from woof.ensemble.request import EnsembleRequest
from woof.ensemble.runtime_context import current_session, ensemble_scope


def request_for_config(path, *, override=None, members=None, keep_member_files=None,
                       recipe=None, trajectories=None, family_weights=None, stochastic=None):
    from woof.config_authority import read_config_authority
    return request_for_payload(read_config_authority(path).payload,
        override=override, members=members, keep_member_files=keep_member_files,
        recipe=recipe, trajectories=trajectories, family_weights=family_weights,
        stochastic=stochastic)


def stochastic_switches(args) -> dict | None:
    """``--sppt/--no-sppt`` and ``--skeb/--no-skeb`` as ``[ensemble.stochastic]`` keys.

    SPPT (Buizza et al. 1999; Palmer et al. 2009) and SKEBS (Shutts 2005;
    Berner et al. 2009, 2011) run WRF's own published scheme
    (``woof.ensemble.stochastic``, WRF v4.6.1 lineage, public domain).
    """
    switches = {}
    if getattr(args, "sppt", None) is not None:
        switches["sppt"] = bool(args.sppt)
    if getattr(args, "skeb", None) is not None:
        switches["skebs"] = bool(args.skeb)
    return switches or None


def add_stochastic_arguments(parser):
    """``woof go`` / ``woof ensemble`` only: the doors whose request carries them (go_cli)."""
    parser.add_argument("--sppt", action=argparse.BooleanOptionalAction, default=None,
                        help="stochastically perturbed parameterization tendencies in every member "
                             "(WRF's SPPT; overrides [ensemble.stochastic] sppt; default: the config's)")
    parser.add_argument("--skeb", action=argparse.BooleanOptionalAction, default=None,
                        help="stochastic kinetic-energy backscatter in every member "
                             "(WRF's SKEBS; overrides [ensemble.stochastic] skebs; default: the config's)")


def request_for_payload(payload, *, override=None, members=None, keep_member_files=None,
                        recipe=None, trajectories=None, family_weights=None, stochastic=None):
    """Use the captured configuration bytes in a fresh forecast worker.

    ``recipe`` and ``trajectories`` are the ``--recipe`` and
    ``--trajectories`` flags; like ``--members`` they win over the table.
    A trajectory list alone selects ``multi-model``, and a multi-model
    request with no member count runs every listed trajectory.
    """
    raw = tomllib.loads(payload.decode("utf-8"))
    from woof.ensemble.calibration_admission import refuse_configured_random
    refuse_configured_random(raw)
    value = raw.get("ensemble") if override is None else override
    if (isinstance(value, int) and not isinstance(value, bool)
            and (recipe is not None or trajectories is not None)):
        value = {"members": value}
    if isinstance(value, dict) and (recipe is not None or trajectories is not None
                                    or value.get("recipe") is not None or value.get("trajectories")):
        value = dict(value)
        if recipe is not None:
            value["recipe"] = recipe
        if trajectories is not None:
            value["trajectories"] = list(trajectories)
        if value.get("trajectories") and value.get("recipe") is None:
            value["recipe"] = "multi-model"
        if value.get("recipe") not in ("multi-model", "member-roster"):
            if recipe is not None and trajectories is None:
                # --recipe time-lagged over a table that listed trajectories.
                value.pop("trajectories", None)
        if members is not None:
            value["members"] = members
        elif "members" not in value and value.get("recipe") == "multi-model":
            value["members"] = len(value.get("trajectories") or ())
        members = None
    elif value is None and (recipe is not None or trajectories is not None):
        chosen = recipe or "multi-model"
        count = members if members is not None else (
            len(trajectories or ()) if chosen == "multi-model" else None)
        if count is None:
            raise ValueError("an ensemble recipe needs a member count: --members N or [ensemble] members")
        value = {"members": count, "recipe": chosen,
                 **({} if trajectories is None else {"trajectories": list(trajectories)})}
        members = None
    if value is None and members is None:
        if keep_member_files is not None:
            raise ValueError("keep member files requires an ensemble member count")
        if stochastic:
            raise ValueError("--sppt/--skeb need an ensemble: --members N or [ensemble] members")
        return None
    request = {} if value is None else EnsembleRequest.from_mapping(value).receipt()
    if stochastic:
        # The flags override the same keys of [ensemble.stochastic]; the rest
        # of a configured stochastic table is kept.
        request["stochastic"] = {**(request.get("stochastic") or {}), **stochastic}
    if members is not None:
        request["members"] = members
    if keep_member_files is not None:
        request["keep_member_files"] = keep_member_files
    if family_weights is not None:
        request["family_weights"] = family_weights
    return EnsembleRequest.from_mapping(request)


def request_for_inputs(*, override=None, members=None, keep_member_files=None):
    """A native input-directory door may inherit the outer run-plan scope."""
    existing = current_session()
    if override is None and existing is not None:
        override = existing.request
    request = request_for_payload(b"", override=override, members=members,
                                  keep_member_files=keep_member_files)
    # An input directory is one trajectory's files, so N > 1 members here
    # would all run them.  Refused before the directory is read.
    from woof.ensemble.member_inputs import refuse_one_input
    refuse_one_input(request, "--wrfinput and --met-em name one trajectory's files, "
                              "so every member would run them.")
    return request


@contextmanager
def production_run_scope(request, *, output_directory, session_factory=None, input_provider=None,
                         restart_roster=None):
    if request is None:
        if restart_roster is not None:
            raise ValueError("--restart-roster requires an ensemble request")
        yield None
        return
    request = EnsembleRequest.from_mapping(request)
    existing = current_session()
    if existing is not None:
        if restart_roster is not None and existing.restart_roster != Path(restart_roster):
            raise ValueError("nested ensemble door has another restart roster")
        if existing.request != request:
            raise ValueError("nested forecast door carries a different ensemble request")
        if input_provider is not None:
            if (existing.member_roster is not None or existing.source_execution is not None
                    or (existing.input_provider is not None and existing.input_provider is not input_provider)):
                raise ValueError("nested recipe forecast already has another member input owner; replacing it would run a different prepared trajectory")
            existing.input_provider = input_provider
        yield existing
        return
    if session_factory is None:
        from woof.ensemble.production import PreparedEnsembleSession
        session_factory = PreparedEnsembleSession
    session = session_factory(request, output_directory=output_directory,
        **({} if input_provider is None else {"input_provider": input_provider}),
        **({} if restart_roster is None else {"restart_roster": restart_roster}))
    with ensemble_scope(session):
        yield session


def add_recipe_arguments(parser):
    """The member-source recipe flags, for the doors that fetch their sources."""
    from woof.ensemble.recipe_door import RECIPES
    parser.add_argument("--family-weights", default=None, metavar="ai=0.5,physics=0.5",
                        help="family probability masses, must sum to 1; defaults to member shares")
    parser.add_argument("--recipe", choices=RECIPES, default=None,
                        help="take each ensemble member from a real source trajectory: "
                             "time-lagged runs earlier cycles of the config's own source "
                             "over the same window, multi-model runs the trajectories "
                             "--trajectories lists, surface-state runs seeded soil "
                             "moisture scales and SST offsets from [ensemble.perturbation], "
                             "member-roster runs named land and fixed surface arms "
                             "from [ensemble.member_variants]")
    parser.add_argument("--trajectories", type=Path, default=None, metavar="FILE",
                        help="the multi-model member list: a JSON or TOML file of "
                             "{source, cycle[, member]} entries, one per member "
                             "(selects --recipe multi-model)")


def add_arguments(parser):
    parser.add_argument("--restart-roster", type=Path, default=None, metavar="JSON",
                        help="continue the exact original members from a durable ensemble restart roster")
    parser.add_argument("--members", type=int, default=None, metavar="N",
                        help="make an N-member ensemble with aggregate products")
    parser.add_argument("--keep-member-files", action="store_true", default=None,
                        help="also retain every member's full history files")
