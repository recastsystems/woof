"""Every member of an N > 1 ensemble runs its own inputs, or the request is refused.

Breakage this module prevents: a door that prepares one trajectory and is
asked for N members hands every member the same prepared inputs.  The run
is N copies of one forecast, and its spread maps (zero everywhere) and
probability maps (0 or 1) are published as an ensemble forecast.

A plain member count (``--members N`` or ``[ensemble] members = N`` with no
recipe) therefore means the automatic choice of
:func:`woof.ensemble.recipes.build_recipe`: the operational ensemble the
source's adapter row declares.  The doors that fetch run it through the
recipe door like any named recipe.  Where the adapter table declares no
runnable ensemble, and at the doors that hold one input, the request is
refused by name with the remedy, before anything is downloaded.

No source is named here.  The adapter table decides.
"""
from __future__ import annotations

from woof.ensemble_admission import copies_breakage, member_source_remedy, one_input_refusal

#: What a plain member count draws its members from, said the same way in
#: every refusal.
AUTOMATIC_CHOICE = "A plain member count takes each member from the source's operational ensemble"


def needs_member_sources(request) -> bool:
    """A plain member count: N > 1, no recipe named and no sources listed.

    A request that lists ``sources`` is not plain: it names each member's
    source, and the session refuses it by that name when no door binds them.
    """
    return (request is not None and request.members > 1
            and request.recipe is None and not request.sources)


def no_automatic_members(source, members, error) -> str:
    """Why this source has no automatic member plan, and what to ask for.

    The recipe door calls this where ``build_recipe`` raised for a plain
    member count.  The adapter row says whether the source declares an
    operational ensemble at all; when it does, the planner's own reason
    (a member set that cannot initialize a run, a window its members do
    not publish, more members than it has) is carried through.
    """
    from woof.source_adapters import get_source_adapter

    try:
        adapter = get_source_adapter(source)
    except (KeyError, ValueError):
        adapter = None
    declared = adapter is not None and bool(
        adapter.member_set or getattr(adapter, "ensemble_source", None))
    reason = (f"that plan cannot be made for this request ({error})" if declared
              else f"{source} declares none")
    return f"{AUTOMATIC_CHOICE}, and {reason}. {member_source_remedy(members)}"


def planned_refusal(request, refusal) -> str:
    """A refused member plan, as a plain member count is told about it.

    Called by :func:`woof.ensemble.recipe_door.plan_recipe` on its own
    refusals, and on nothing else: the door's later gates (the card,
    memory, geography, the renderer, the disk) are not about where the
    members come from and keep their own sentences.

    A named recipe keeps the plan's own sentence.  A plain member count
    leads with what the request would otherwise have been: N copies.  The
    plan's sentence follows word for word.
    """
    text = str(refusal).strip()
    if not needs_member_sources(request):
        return text
    lead = copies_breakage(request.members)
    if text.startswith(AUTOMATIC_CHOICE):
        return f"{lead} {text}"
    return (f"{lead} Each member needs its own source trajectory instead, and that "
            f"plan was refused: {text}")


def refuse_one_input(request, holds) -> None:
    """Refuse N > 1 members at a door that holds one input.

    ``holds`` is the door's clause for that one input (a prepared bundle,
    a checkpoint, an input directory).  Listed sources are refused in
    their own words, since no such door binds them.  A named recipe is
    left to the session, which refuses it in the recipe's own words.
    """
    if request is None or request.recipe is not None:
        return
    if request.sources:
        from woof.ensemble_admission import unbound_sources_refusal
        raise ValueError(unbound_sources_refusal())
    # This existing initializer gives each member its own seeded wind
    # state in PreparedEnsembleSession. It consumes one source by design;
    # refusing it here made the native-input door unreachable even though
    # the complete ordinary member runner already owns the initialization.
    from woof.ensemble.initial_perturbation import (
        is_initial_perturbation, validate_initial_perturbation)
    if is_initial_perturbation(request.perturbation):
        validate_initial_perturbation(request.perturbation)
        return
    if request.members > 1:
        raise ValueError(one_input_refusal(request.members, holds))


def config_request(config, *, members=None, keep_member_files=None):
    """The ensemble request a config file and the member flags make, or None."""
    from pathlib import Path

    if config is None or not Path(config).is_file():
        if members is None:
            return None
        from woof.ensemble.request import EnsembleRequest
        return EnsembleRequest.from_mapping(members)
    from woof.ensemble.door import request_for_config
    return request_for_config(config, members=members, keep_member_files=keep_member_files)


__all__ = ["AUTOMATIC_CHOICE", "config_request", "needs_member_sources", "no_automatic_members",
           "one_input_refusal", "planned_refusal", "refuse_one_input"]
