"""Data-driven GPUWM physics inventory and per-domain plan validation.

This module is intentionally usable with the Python standard library alone.
It describes executable components; it does not import or initialize NumPy,
CuPy, CUDA, forecast drivers, or physics implementations.  The v2 contract is
additive: existing v1 launchers continue to own execution until a caller
explicitly consumes a validated v2 plan.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


REGISTRY_SCHEMA = "gpuwm-physics-registry-v2"
PLAN_SCHEMA = "gpuwm-physics-plan-v2"
VALIDATION_SCHEMA = "gpuwm-physics-plan-validation-v2"

#: Issue codes that are a question about THE MACHINE this report was
#: produced on, not about the plan.
#:
#: A plan is portable: the same document is valid on the laptop that
#: authored it and on the node that runs it, and that is the property
#: ``launchable`` states.  Whether a 225 MB dataset or a table set is
#: INSTALLED HERE is a different question with a different answer per
#: machine, and it is answered by the door that is about to load the
#: scheme -- Thompson's staging preflight and loader for a table set,
#: :func:`woof.config.validate_experiment_preparation` for the aerosol
#: lateral-forcing dataset -- each naming the member that is missing and
#: the command that stages it.
#:
#: So plan review REPORTS them, in ``install_state``, and leaves
#: ``launchable`` to the plan.  Appending them to ``errors`` instead made
#: a DEFAULT template unlaunchable on every fresh install, because two of
#: classic Thompson's four tables are excluded from the wheel and arrive
#: only through ``woof fetch-tables``: ``woof source
#: --validate-physics-plan`` exited nonzero on a correct plan, and the
#: shipped suites went red in exactly the state the clean-venv release
#: replay runs in.  Collecting the requirements and never resolving them
#: (which is what this replaced) is the opposite defect and is not
#: restored: the resolution is reported, with what is missing and where
#: it was looked for.
INSTALL_STATE_CODES = frozenset({
    "asset-unresolved",
    "lateral-forcing-dataset",
})

WSM6_TEMPLATE_ID = "wsm6-ysu-mm5-noah-no-radiation-v1"
THOMPSON_TEMPLATE_ID = "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1"
THOMPSON_KF_TEMPLATE_ID = "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1"
MORRISON_TEMPLATE_ID = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
NSSL2_TEMPLATE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1"
)
NSSL2_LEGACY_RRTMG_TEMPLATE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1"
)

# Historical public identifiers remain inputs. Menus emit the descriptive IDs.
TEMPLATE_ID_ALIASES = {
    "thompson-mp8-ysu-mm5-noah-validation-v1": THOMPSON_TEMPLATE_ID,
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1": NSSL2_TEMPLATE_ID,
    "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1": NSSL2_LEGACY_RRTMG_TEMPLATE_ID,
}

#: The registered template whose suite `woof domain` emits by default
#: (product decision, 2026-07-29): Thompson mp8 in the certified
#: real-data set.  Morrison stays registered and fully selectable at its
#: own maturity label; a test binds the wizard's emitted selectors to
#: this template's components so the two cannot drift apart.
DEFAULT_TEMPLATE_ID = THOMPSON_KF_TEMPLATE_ID

_PINNED_REGISTRY_ENV = "WOOF_PINNED_PHYSICS_REGISTRY"
_PINNED_REGISTRY_SHA256_ENV = "WOOF_PINNED_PHYSICS_REGISTRY_SHA256"
_pinned_path = os.environ.get(_PINNED_REGISTRY_ENV)
_pinned_sha256 = os.environ.get(_PINNED_REGISTRY_SHA256_ENV)
if (_pinned_path is None) != (_pinned_sha256 is None):
    raise RuntimeError(
        "pinned physics registry path and SHA-256 must be supplied together")
_REGISTRY_PATH = Path(
    _pinned_path or Path(__file__).with_name("physics_registry_v2.json")
).resolve()
try:
    _REGISTRY_BYTES = _REGISTRY_PATH.read_bytes()
except OSError as exc:
    raise RuntimeError(
        f"GPUWM physics registry is unreadable: {_REGISTRY_PATH}") from exc
if _pinned_sha256 is not None:
    if (len(_pinned_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in _pinned_sha256)
            or hashlib.sha256(_REGISTRY_BYTES).hexdigest()
            != _pinned_sha256):
        raise RuntimeError(
            "pinned physics registry raw SHA-256 differs before JSON decode")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def canonical_maturity(
    maturity: object, registry: Mapping[str, object] | None = None,
) -> object:
    """Read a historical maturity spelling as its current evidence label.

    The generated ladder owns the aliases as well as the canonical names.
    Unknown values remain unknown; an alias never adds an evidence rung.
    Registry bytes and profile identifiers are not rewritten on read.
    """

    if not isinstance(maturity, str):
        return maturity
    selected = _REGISTRY if registry is None else registry
    ladder = selected.get("maturity_ladder", {})
    aliases = ladder.get("aliases", {}) if isinstance(ladder, dict) else {}
    return aliases.get(maturity, maturity) if isinstance(aliases, dict) else maturity


def canonical_template_id(
    template_id: object, registry: Mapping[str, object] | None = None,
) -> object:
    """Resolve a historical profile ID without changing its physics."""

    if not isinstance(template_id, str):
        return template_id
    selected = _REGISTRY if registry is None else registry
    aliases = selected.get("template_aliases", {})
    return aliases.get(template_id, template_id) if isinstance(aliases, dict) else template_id


def iter_maturity_surfaces(registry: Mapping[str, object]):
    """Yield ``(path, maturity)`` for every surface carrying a maturity.

    The walk is structural rather than a list of known surfaces, so a
    maturity introduced on a surface nobody thought of is still checked.
    ``maturity_ladder`` and ``evidence_axes`` are the vocabulary itself and
    are skipped: their rung names are definitions, not uses.
    """

    def walk(node: object, path: str):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "maturity" and isinstance(value, str):
                    yield f"{path}.maturity", value
                else:
                    yield from walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, f"{path}[{index}]")

    for key, value in registry.items():
        if key in ("maturity_ladder", "evidence_axes"):
            continue
        yield from walk(value, key)


def _enforce_evidence_axes(registry: Mapping[str, object]) -> None:
    """Load-time two-axis membership enforcement (owner decision D-22).

    Enforced here, at the loader, because the repository's own ethos is
    that a malformed document does not get to reach a caller.  What is
    enforced is MEMBERSHIP on both axes -- every maturity value at every
    surface names a rung, every implemented option's scientific_evidence
    names an enum entry -- which cannot take a working template out of
    service: a template only fails it by carrying a word the vocabulary
    does not define.  The composition rule is deliberately NOT enforced
    here (D-16): clause C2 flags six shipped templates, and a blocking
    loader would take WSM6 and both NSSL-2 profiles out of service, so it
    is a registry-document invariant that a test enforces instead.
    """

    ladder = registry.get("maturity_ladder")
    if not isinstance(ladder, dict):
        raise RuntimeError("bundled GPUWM physics registry lacks the "
                           "maturity_ladder block")
    order = ladder.get("order")
    rungs = ladder.get("rungs")
    if not isinstance(order, list) or not isinstance(rungs, dict):
        raise RuntimeError("maturity_ladder must carry an order list and a "
                           "rungs map")
    if list(rungs) and set(order) != set(rungs):
        raise RuntimeError(
            "maturity_ladder.order and maturity_ladder.rungs disagree: "
            f"{sorted(set(order) ^ set(rungs))}")
    aliases = ladder.get("aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(alias, str) or alias in rungs
        or not isinstance(target, str) or target not in rungs
        for alias, target in aliases.items()
    ):
        raise RuntimeError(
            "maturity_ladder.aliases must map historical spellings to "
            "canonical rungs without replacing a rung")

    axes = registry.get("evidence_axes")
    if not isinstance(axes, dict):
        raise RuntimeError("bundled GPUWM physics registry lacks the "
                           "evidence_axes block")
    for axis_key in ("maturity", "scientific"):
        if not isinstance(axes.get(axis_key), dict):
            raise RuntimeError(f"evidence_axes lacks the {axis_key!r} axis")
    if axes.get("conformance_implies_scientific_validation") is not False:
        raise RuntimeError(
            "evidence_axes.conformance_implies_scientific_validation must be "
            "present and false: agreement with WRF is agreement with a "
            "model and never implies scientific validation")
    if axes["maturity"].get("rungs") != rungs:
        raise RuntimeError(
            "evidence_axes.maturity.rungs and maturity_ladder.rungs are two "
            "orderings of the same names; they must be identical")

    in_use = set()
    for path, maturity in iter_maturity_surfaces(registry):
        maturity = canonical_maturity(maturity, registry)
        if maturity not in rungs:
            raise RuntimeError(
                f"{path} carries maturity {maturity!r}, which names no rung "
                f"of maturity_ladder; rungs are {sorted(rungs)}")
        in_use.add(maturity)
    if in_use != set(rungs):
        raise RuntimeError(
            "maturity_ladder.rungs must equal the maturity values in use; "
            f"unused rungs {sorted(set(rungs) - in_use)}, "
            f"undeclared values {sorted(in_use - set(rungs))}")

    scientific_enum = axes["scientific"].get("enum")
    if not isinstance(scientific_enum, dict) or not scientific_enum:
        raise RuntimeError("evidence_axes.scientific.enum is missing")
    components = registry.get("components")
    for component_id, component in (components or {}).items():
        if not isinstance(component, dict):
            continue
        for option_id, option in component.get("options", {}).items():
            if not isinstance(option, dict) or option.get(
                    "implemented") is not True:
                continue
            value = option.get("scientific_evidence")
            if value not in scientific_enum:
                raise RuntimeError(
                    f"components.{component_id}.options.{option_id}"
                    f".scientific_evidence is {value!r}, which names no "
                    f"entry of evidence_axes.scientific.enum")

    policy = registry.get("warning_policy")
    policy = policy if isinstance(policy, dict) else {}
    for tier, key in (("nonwarning", "nonwarning_maturities"),
                      ("warn", "warn_maturities")):
        computed = [
            name for name in order
            if isinstance(rungs.get(name), dict)
            and rungs[name].get("warning_tier") == tier
        ]
        if policy.get(key) != computed:
            raise RuntimeError(
                f"warning_policy.{key} is {policy.get(key)!r} but the ladder "
                f"computes {computed!r}; the ladder is the single source")


def _load_registry() -> dict[str, object]:
    try:
        text = _REGISTRY_BYTES.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("GPUWM physics registry is not UTF-8") from exc
    value = json.loads(text, parse_constant=_reject_json_constant)
    if not isinstance(value, dict) or value.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("bundled GPUWM physics registry v2 schema drift")
    if value.get("plan_schema") != PLAN_SCHEMA:
        raise RuntimeError("bundled GPUWM physics plan schema drift")
    if value.get("validation_schema") != VALIDATION_SCHEMA:
        raise RuntimeError("bundled GPUWM physics validation schema drift")
    _enforce_evidence_axes(value)
    return value


_REGISTRY = _load_registry()

#: The one ordering of maturity names in this module, read from the
#: generated document rather than restated here.  Every consumer that needs
#: to compare two maturities -- the warning tiers below, the composition
#: ceiling in tests/test_physics_registry_composition.py -- reads this.
MATURITY_RANK: dict[str, int] = {
    name: rank
    for rank, name in enumerate(_REGISTRY["maturity_ladder"]["order"])
}


def maturity_rank(maturity: object) -> int | None:
    """Rank of a current or historical maturity, or None if unknown."""

    maturity = canonical_maturity(maturity)
    return MATURITY_RANK.get(maturity) if isinstance(maturity, str) else None


def _ladder_tier(registry: Mapping[str, object], tier: str) -> list[str]:
    """Return the ladder-ordered maturities in one warning tier."""

    ladder = registry.get("maturity_ladder")
    ladder = ladder if isinstance(ladder, dict) else {}
    order = ladder.get("order")
    rungs = ladder.get("rungs")
    if not isinstance(order, list) or not isinstance(rungs, dict):
        return []
    return [
        name for name in order
        if isinstance(rungs.get(name), dict)
        and rungs[name].get("warning_tier") == tier
    ]


def canonical_json(value: object) -> str:
    """Return the one byte-stable JSON rendering used by registry hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Hash canonical UTF-8 JSON rather than Python insertion order."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


#: The registry object this module loaded, serialised once in document order
#: with its canonical hash.  The registry is JSON-origin data
#: (:func:`_load_registry`), so re-parsing that text gives the same
#: independent dicts, lists and scalars, in the same order, as a ``deepcopy``
#: -- at a fraction of the cost.  Measured on one cell of
#: tests/test_physics_combination_matrix.py (1,429 tuples): the 2,104
#: deepcopies of the document were the bulk of the cell, which is what left
#: one xdist worker on one core after the other seven had finished.  The
#: fast path applies only while ``_REGISTRY`` is still this loaded object:
#: a test or tool that installs an edited document in its place, and may go
#: on editing it, gets the original deepcopy and per-call hash.
_BUILTIN_REGISTRY = _REGISTRY
_BUILTIN_REGISTRY_TEXT = json.dumps(_REGISTRY, ensure_ascii=False, allow_nan=False)
_BUILTIN_REGISTRY_SHA256: str | None = None


def physics_registry() -> dict[str, object]:
    """Return an independent copy of the immutable built-in registry."""

    if _REGISTRY is _BUILTIN_REGISTRY:
        return json.loads(_BUILTIN_REGISTRY_TEXT, parse_constant=_reject_json_constant)
    return deepcopy(_REGISTRY)


def registry_sha256(registry: Mapping[str, object] | None = None) -> str:
    """Return the semantic hash of the selected registry document."""

    global _BUILTIN_REGISTRY_SHA256
    if registry is not None:
        return canonical_sha256(registry)
    if _REGISTRY is not _BUILTIN_REGISTRY:
        return canonical_sha256(_REGISTRY)
    if _BUILTIN_REGISTRY_SHA256 is None:
        # The built-in document never changes in-process; hash it once.
        _BUILTIN_REGISTRY_SHA256 = canonical_sha256(_BUILTIN_REGISTRY)
    return _BUILTIN_REGISTRY_SHA256


# ------------------------------------------------ physics identity (A153)
#: What a physics selection binds of this registry.
#:
#: The document digest (:func:`registry_sha256`) moves with every byte of
#: the document, and most of the document is prose: warnings, notes and
#: reasons that cite ``file:line`` and are re-pinned whenever the cited
#: code moves.  A selection receipt that bound the document digest refused
#: every earlier preparation as "physics selection differs" after a
#: citation-only commit, while the forecast it would have run was byte
#: identical (A153: the Shin-Hong ``kpbl < kte`` guard citation and the
#: LES ``km_opt`` header citations refused the default, NSSL, P3, KF,
#: Noah-MP and Shin-Hong selections across 2.8.0 -> 2.8.1).  It also bound
#: every OTHER scheme: a change to WDM6's restart identity refused a
#: Thompson preparation.
#:
#: So a selection binds the registry's physics PARTS it runs on, each with
#: its documentation and its admission rules removed and nothing else
#: removed:
#:
#: ``document``
#:     ``schema`` and ``registry_version``, the format the rest is read in.
#: ``components.<component>``
#:     every component's selector keys and per-domain selector rules,
#:     because they decide which option a configuration resolves to.
#: ``components.<component>.options.<option>``
#:     each SELECTED option: selectors, parameters (the option's own
#:     identity values, the Noah-MP ``opt_*`` and MYNN ``bl_mynn_*`` pins
#:     among them), the constraints the preflight cannot re-check (see
#:     :data:`REGISTRY_CONSTRAINTS_KEPT_IN_PHYSICS_IDENTITY`), implemented
#:     flag, reachability state, table assets and their SHA-256s, consumer
#:     rows (restart algorithm identity, callable class, moment and field
#:     inventories) and extensions.  This is where a changed option, table
#:     or kernel identity moves the digest.
#: ``parameters.<knob>``
#:     each IMPLEMENTED knob's type, default, bounds and enum, one part per
#:     knob so a refusal names the knob.  A default is read by any scheme
#:     whose configuration omits the knob, so a knob is in every
#:     selection's scope, except a knob the registry gives to one component
#:     (``component_id``) that only that component's unselected options
#:     read (see :func:`registry_knob_readers`).  A knob the registry
#:     declares but does not implement (:func:`parameter_is_implemented`)
#:     is no part: no configuration can set it, so it has no physics, and
#:     implementing it later is an addition to the parts.  A knob whose
#:     declaration says it is read only under other settings
#:     (``read_when``, see :func:`registry_knob_is_read`) is still a part,
#:     because which settings hold is a property of a configuration, not
#:     of the registry; it is the comparison against a preparation made
#:     before the knob existed that reads the condition.
#: ``transitions``
#:     the nest microphysics transitions.
#: ``templates.<profile>``
#:     the named profile's components and parameters, when a profile is
#:     named, less the entries at their off values (see
#:     :func:`registry_off_template`), since a template that states a
#:     component's off option or a knob's off value composes what one that
#:     omits it does.
#:
#: The receipt records one digest per part, so a refusal names the part
#: that changed.  v1 bound each option's constraints whole and the
#: parameters block as one part: a new PBL admitted beside GF and the MYNN
#: surface layer refused every 2.8.0 preparation of either, and a knob
#: added for one scheme refused every selection, naming only "parameters".
#: v2 bound every declared knob and every template entry: off-by-default
#: physics (an urban component whose ``none`` option every template names,
#: two knobs only its other options read, two radiation knobs implemented
#: at their off value) refused all 56 of 2.8.0's receipts by component,
#: knob and template, 54 of them for physics that runs only when selected.
#:
#: What a preparation's registry lacked is resolved by the comparison
#: (:func:`woof.physics_compat.physics_selection_differences`): a
#: component, option or knob this build added counts as equal when the
#: configuration resolves it to its off value (:func:`setting_off_value`,
#: :func:`component_off_option`), and is refused, named, when it does not.
#: What is outside the parts, by top-level block and why:
# v4 excludes WRF citation and refusal prose added to the v3 inventory.
# Older receipt parts resolve through their retained registry document.
REGISTRY_PHYSICS_IDENTITY_SCHEMA = "gpuwm-physics-registry-parts-v4"

#: Top-level blocks no selection's physics identity binds.
REGISTRY_BLOCKS_OUTSIDE_PHYSICS_IDENTITY: Mapping[str, str] = {
    "analysis_nudging": "FDDA capability documentation; selected run controls, native input authority and compiled kernels bind execution",
    "diagnostic_packages": "passive package descriptions; selected parameters and compiled kernels bind the diagnostic identity",
    "diagnostic_state": "WRF diagnostic field metadata; checkpoint inventory binds the actual fields",
    "template_aliases": "historical profile spellings, not executable settings",
    "authority": (
        "the registry's declarations of its own contracts and the WRF "
        "v4.6.1 compatibility matrix: statements about the document and "
        "about WRF.  What they decide for a selection, its resolved "
        "components and its acknowledgement, the receipt records and "
        "compares itself"),
    "runner_routes": (
        "admission: which templates, options and keys each runner offers. "
        "Its outcome for a selection is the receipt's governance record, "
        "compared field by field"),
    "plan_binding": (
        "the physics-plan document's binding contract, read by plan review "
        "and never by a forecast"),
    "plan_schema": "the plan document's schema id, read by plan review",
    "validation_schema": "the plan review report's schema id",
    "evidence_axes": (
        "the conformance and scientific-evidence vocabularies: how far an "
        "option has been checked, not what it computes"),
    "maturity_ladder": (
        "the conformance rungs and the template composition rule: evidence "
        "bookkeeping a registry test enforces"),
    "warning_policy": (
        "which maturity tiers print a warning; it decides what is said "
        "about a run, never what runs (maturity_never_blocks)"),
    "system_suites": (
        "rows describing another forecast system's suites in this engine's "
        "terms (woof/system_suites.py): no template, route or default "
        "resolves to one, so they decide nothing a selection runs; a "
        "profile that imitates a suite carries the settings itself and is "
        "bound through them"),
    "templates": (
        "bound per part: only the profile a receipt names "
        "(templates.<profile>)"),
    "components": "bound per part (see above)",
    "parameters": "bound per knob (parameters.<knob>)",
    "transitions": "bound as the transitions part",
    "schema": "bound in the document part",
    "registry_version": "bound in the document part",
}

#: Evidence records nested inside an option, dropped whole wherever they
#: hold an object.
REGISTRY_EVIDENCE_RECORDS: Mapping[str, str] = {
    "column_oracle_evidence": (
        "a scheme's recorded column-oracle and trajectory comparison "
        "(fixtures, gate constants, verdicts, measured numbers): the "
        "verification of the scheme, not the scheme"),
    "measured_forecast_sensitivity": (
        "a measured sensitivity snapshot and its test pointer, which the "
        "registry itself calls a snapshot and not a physics pin"),
}

#: Fields that hold documentation wherever their value is prose (a
#: string, a list of strings, or null).  A field of the same name that
#: holds an object is walked, not dropped.
REGISTRY_DOCUMENTATION_FIELDS: Mapping[str, str] = {
    "wrf_reference": "a WRF source citation",
    "rte_rrtmgp_refusal": "the words explaining the bound cloud-optics coupling refusal",
    "implementation_origin": "implementation provenance for display; no numerical consumer reads it",
    # User-facing caveats.  They cite file:line and are re-pinned when the
    # cited code moves; both A153 cases were edits here.
    "warnings": "user-facing caveats with file:line citations",
    # Citations and provenance: where WRF or this package does a thing.
    "source": "a citation (a WRF Registry line, a table's provenance)",
    "wrf_source": "a WRF file:line citation",
    "wrf_authority": "a WRF citation",
    "wrf_call_site": "a WRF file:line citation",
    "wrf_coefficient_policy": "a WRF file:line citation",
    "wrf_flag_source": "a WRF file:line citation",
    "wrf_live_consumer": "a WRF file:line citation",
    "wrf_package": "a WRF Registry citation",
    "wrf_counterpart": "the WRF counterpart named in words",
    "gpuwm_runtime_source": "a citation of this package's own source",
    "gpuwm_evidence": "a citation of this package's own source",
    "gpuwm_implementation": "a path::routine pointer",
    "gpuwm_implementation_evidence": "a test pointer",
    "production_call_site": "a path::routine pointer",
    "production_call_site_note": "prose about that call site",
    "consuming_read": "the file that reads a knob; it moves with the code",
    "call_site_pin": "a test pointer",
    "installed_state_pin": "a test pointer",
    "validator": "a path::routine pointer",
    "authority": "a citation in words",
    "classification": "prose classifying a requirement",
    # The words of a decision whose value a sibling field carries.
    "reason": "the words of a constraint or refusal",
    # (No unimplemented_reason row: only a knob the registry does not
    # implement carries one, and such a knob is no part at all.)
    "clear_air_floor_reason": "clear_air_floor_status carries the fact",
    "reflectivity_route_reason": "reflectivity_route carries the fact",
    "reflectivity_native_reason": "the reflectivity route carries the fact",
    "blocker": "reachability.state carries the fact",
    "basis": "the basis of a claim in words",
    "consequence": "the consequence of a claim in words",
    "restart_identity_binding": "prose about where an identity is bound",
    # Descriptive prose and display text.
    "note": "an explanatory note",
    "withheld_aerosol_number_note": "an explanatory note",
    "aerosol_number_mixing_note": "an explanatory note",
    "evidence": "the evidence for a claim in words",
    "behaviour": "a measured behaviour in words",
    "measured": "a measured result in words",
    "execution_model": "where a scheme runs, in words",
    "fixed_mode": "a mode described in words",
    "droplet_activation": "prose citing the native activation and sedimentation routines",
    "default_lane": "a mode described in words",
    "activation": "the selectors that activate an extension, in words",
    "activation_bin_edge_policy": "a policy described in words",
    "ordering": "an ordering described in words",
    "ported_operators": "the ported operators named in words",
    "shipped_consequence": "prose",
    "shipped_profile": "prose",
    "operator_workaround": "prose",
    "removed_constraint": "prose about a constraint no longer present",
    "label": "a display name",
    # Evidence labels; the vocabularies they name are outside the parts.
    "maturity": "a conformance evidence label",
    "scientific_evidence": "a scientific evidence label",
    "verification_scope": "the scope of a reported evidence label",
    "verification_scope_note": "the dated limits of the evidence",
}

#: Objects whose KEYS are physics and whose values are prose: the key set
#: stays bound (which settings a constraint names, which siblings a family
#: refuses) and the words go.
REGISTRY_DOCUMENTATION_VALUE_MAPS: Mapping[str, str] = {
    "family_siblings_refused": "the refused sibling ids stay bound",
    "unported_wrf_siblings": "the unported sibling ids stay bound",
}

#: Constraint kinds (keys of an option's ``constraints``) that are
#: ADMISSION and leave the parts, for the reason ``runner_routes`` is
#: outside them: they decide whether a configuration is admitted, never
#: what an admitted one computes.  The preflight recomputes the receipt
#: from the prepared configuration through
#: ``woof.physics_compat.validate_physics_capabilities``, which reads all
#: three, so a rule that no longer admits the prepared configuration
#: refuses there, naming the rule (a named profile raises; a per-domain
#: receipt records the blocker, and ``registry_blocker`` and ``components``
#: are compared).  A rule that still admits it leaves its physics alone:
#: 2.8.1's UW PBL joining GF's and the MYNN surface layer's admitted PBLs
#: refused 18 of 2.8.0's 56 receipts while v1 bound them.  An option's own
#: identity values that ``required_settings`` repeats stay bound through
#: its ``parameters`` block, which carries the same values.
REGISTRY_ADMISSION_CONSTRAINTS: Mapping[str, str] = {
    "requires_components": (
        "which options of another component this one runs beside"),
    "requires_components_reasons": "the words of requires_components",
    "forbidden_setting_values_reasons": "the words of forbidden_setting_values",
    "required_settings": (
        "the single value a setting must hold; the option's own values are "
        "repeated in its parameters block, and the rest (moist, km_opt, "
        "khdif, kvdif, bl_pbl_physics, num_soil_layers) are settings the "
        "prepared configuration carries"),
    "required_settings_reasons": "the words of required_settings",
    "admitted_setting_values": "the values a setting may hold",
    "admitted_setting_values_reasons": (
        "the words of admitted_setting_values"),
}

#: Constraint kinds that STAY in an option's part, each because the
#: preflight's recomputation does not re-check it, so the identity is the
#: only place a change to it reaches a prepared receipt.
REGISTRY_CONSTRAINTS_KEPT_IN_PHYSICS_IDENTITY: Mapping[str, str] = {
    "admitted_setting_ranges": "validate_physics_capabilities does not recheck these range bounds",
    "forbidden_setting_values": (
        "read by plan review (validate_physics_plan) only; "
        "validate_physics_capabilities does not evaluate it"),
    "refused_when": (
        "a rule with a sources clause fires only at plan review and the "
        "preparation door, never in the preflight's recomputation, so those "
        "rules stay bound; a rule without one is evaluated by "
        "validate_physics_capabilities like the admission kinds above and "
        "leaves the part with them (_admission_conditional_refusal)"),
}


def _is_prose(value: object) -> bool:
    return value is None or isinstance(value, str) or (
        isinstance(value, list)
        and all(isinstance(item, str) for item in value))


def _admission_conditional_refusal(rule: object) -> bool:
    """Is this ``refused_when`` rule one the preflight's recomputation
    re-checks, and so admission rather than physics?

    ``woof.physics_compat.validate_physics_capabilities`` evaluates every
    well-formed rule (:func:`_conditional_refusals`) against the prepared
    configuration, and a rule that no longer admits it refuses there by
    name, exactly as the kinds in :data:`REGISTRY_ADMISSION_CONSTRAINTS`
    do.  The exception is a rule with a ``sources`` clause: a RunConfig
    carries no source id, so :func:`_conditional_refusal_fires` never
    fires it there, and such a rule stays bound.  A malformed rule fires
    nowhere and stays bound, so an edit that repairs it is still seen.

    Bound per kind, a rule of this kind refused every 2.8.0 and 2.8.1
    preparation of the 1.5-order TKE closure on the 2.8.2 integration
    line, once development branch moved the closure's PBL-off
    requirement out of ``required_settings`` into a ``diff_opt = 2``
    rule: the closure computes what it did, and the preflight still
    refuses every configuration the rule refuses.
    """

    return (isinstance(rule, Mapping)
            and bool(_conditional_refusals({"refused_when": [rule]}))
            and not isinstance(rule.get("sources"), list))


def _physics_constraints(constraints: Mapping[str, object]) -> dict:
    """An option's ``constraints`` as its physics part binds them."""

    kept: dict[str, object] = {}
    for name, rule in constraints.items():
        if name in REGISTRY_ADMISSION_CONSTRAINTS:
            continue
        if name == "refused_when" and isinstance(rule, list):
            rule = [item for item in rule
                    if not _admission_conditional_refusal(item)]
            if not rule:
                continue
        kept[name] = rule
    return kept


def strip_registry_documentation(node: object) -> object:
    """``node`` with every documentation field of the tables above removed,
    and every admission constraint (:data:`REGISTRY_ADMISSION_CONSTRAINTS`,
    and each ``refused_when`` rule the preflight re-checks) removed from a
    ``constraints`` object."""

    if isinstance(node, Mapping):
        kept: dict[str, object] = {}
        for key, value in node.items():
            if (key == "compatible_previous_enums" and isinstance(value, list)
                    and all(isinstance(previous, list) for previous in value)):
                # Admission history is not the current physical declaration.
                # The comparator verifies the complete historical hash and
                # the selected value before applying an enum-only extension.
                continue
            if (key == "compatible_previous_forbidden_settings" and isinstance(value, list)
                    and all(isinstance(previous, Mapping) for previous in value)):
                continue
            if (key in REGISTRY_EVIDENCE_RECORDS
                    and isinstance(value, Mapping)):
                continue
            if key == "constraints" and isinstance(value, Mapping):
                value = _physics_constraints(value)
            if key in REGISTRY_DOCUMENTATION_FIELDS and _is_prose(value):
                continue
            if (key in REGISTRY_DOCUMENTATION_VALUE_MAPS
                    and isinstance(value, Mapping)
                    and all(_is_prose(item) for item in value.values())):
                kept[key] = sorted(value)
                continue
            kept[key] = strip_registry_documentation(value)
        return kept
    if isinstance(node, list):
        return [strip_registry_documentation(item) for item in node]
    return node


#: A setting this build has no single off value for (see
#: :func:`setting_off_value`).
NO_OFF_VALUE = object()

_RUN_CONFIG_DEFAULTS: dict[str, object] | None = None


def _run_config_defaults() -> Mapping[str, object]:
    """What a configuration that states no value for a setting runs with:
    a ``RunConfig`` built from its own defaults, as a loaded TOML that
    omits the setting is."""

    global _RUN_CONFIG_DEFAULTS
    if _RUN_CONFIG_DEFAULTS is None:
        from dataclasses import MISSING, fields

        from woof.config import RunConfig

        built = RunConfig(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, ztop=1.0,
                          dt=1.0, run_seconds=1.0)
        _RUN_CONFIG_DEFAULTS = {
            field.name: getattr(built, field.name)
            for field in fields(RunConfig)
            if field.default is not MISSING
            or field.default_factory is not MISSING}
    return _RUN_CONFIG_DEFAULTS


def same_setting_value(left: object, right: object) -> bool:
    """Equal as setting values: a boolean never equals a number."""

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same_setting_value(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def setting_off_value(name: str,
                      registry: Mapping[str, object] | None = None) -> object:
    """The value this build runs a setting at when a configuration omits it.

    ``RunConfig``'s default, or the registry knob's declared ``default``
    where ``RunConfig`` has no such field; :data:`NO_OFF_VALUE` when
    neither declares one, or when both do and disagree (then which of the
    two a configuration gets depends on the door, so neither is off).
    Physics added off by default is off at this value, which is what lets
    a preparation made before it existed resolve (A153).
    """

    selected = physics_registry() if registry is None else registry
    parameters = selected.get("parameters")
    spec = (parameters.get(name) if isinstance(parameters, Mapping)
            else None)
    declared = (spec.get("default", NO_OFF_VALUE)
                if isinstance(spec, Mapping) else NO_OFF_VALUE)
    run = _run_config_defaults().get(name, NO_OFF_VALUE)
    if run is NO_OFF_VALUE:
        return declared
    if declared is not NO_OFF_VALUE and not same_setting_value(
            declared, run):
        return NO_OFF_VALUE
    return run


def component_off_option(component_id: str,
                         registry: Mapping[str, object] | None = None
                         ) -> str | None:
    """The option of a component a configuration stating none of its
    selectors resolves to: the one option whose selectors all sit at their
    :func:`setting_off_value`, or ``None`` when no single option does."""

    selected = physics_registry() if registry is None else registry
    component = (selected.get("components") or {}).get(component_id)
    options = (component.get("options")
               if isinstance(component, Mapping) else None)
    if not isinstance(options, Mapping):
        return None
    matched = [
        option_id for option_id, option in options.items()
        if isinstance(option, Mapping)
        and isinstance(option.get("selectors"), Mapping)
        and option["selectors"]
        and all(same_setting_value(setting_off_value(key, selected), value)
                for key, value in option["selectors"].items())]
    return matched[0] if len(matched) == 1 else None


def registry_knob_readers(knob: str,
                          registry: Mapping[str, object] | None = None
                          ) -> frozenset[str] | None:
    """The options that read a knob the registry gives to one component.

    ``None`` unless the knob's declaration names its ``component_id``:
    only then does the registry say which code reads it, and a knob
    without one may be read anywhere.  Otherwise the options of that
    component that name the knob in their parameters or in a setting rule
    of their constraints.  A selection that selects none of them does not
    read the knob (an empty set means the component reads it whatever its
    option, so it is read whenever the component is selected).
    """

    selected = physics_registry() if registry is None else registry
    spec = (selected.get("parameters") or {}).get(knob)
    owner = spec.get("component_id") if isinstance(spec, Mapping) else None
    if not isinstance(owner, str):
        return None
    component = (selected.get("components") or {}).get(owner)
    options = (component.get("options")
               if isinstance(component, Mapping) else None)
    readers = set()
    for option_id, option in (options or {}).items():
        if not isinstance(option, Mapping):
            continue
        rules = option.get("constraints")
        rules = rules if isinstance(rules, Mapping) else {}
        if knob in (option.get("parameters") or {}) or any(
                isinstance(rule, Mapping) and knob in rule
                for rule in rules.values()):
            readers.add(f"{owner}.{option_id}")
    return frozenset(readers)


def registry_knob_is_read(knob: str, settings: Mapping[str, object] | object,
                          registry: Mapping[str, object] | None = None
                          ) -> bool:
    """Whether a configuration reads a knob at all.

    A knob's declaration may name the settings it is read under,
    ``read_when: {setting: value, ...}``, every one of which must hold (a
    tile count read only with tiles on). ``read_when_any`` is an OR
    of these condition rows, in addition to ``read_when``. A setting the configuration
    omits holds its :func:`setting_off_value`.  ``True`` when the
    declaration names no condition, when every condition holds, or when an
    omitted setting has no single off value (then it cannot be shown
    unread).  ``settings`` is a mapping or an object with the settings as
    attributes (a ``RunConfig``).
    """

    selected = physics_registry() if registry is None else registry
    parameters = selected.get("parameters")
    spec = (parameters.get(knob) if isinstance(parameters, Mapping)
            else None)
    if not isinstance(spec, Mapping):
        return True
    absent = object()

    def holds(condition):
        if not isinstance(condition, Mapping) or not condition:
            return True
        for name, wanted in condition.items():
            value = (settings.get(name, absent) if isinstance(settings, Mapping)
                     else getattr(settings, name, absent))
            if value is absent:
                value = setting_off_value(str(name), selected)
                if value is NO_OFF_VALUE:
                    continue  # Unknown cannot prove this reader inactive.
            if not same_setting_value(value, wanted):
                return False
        return True

    alternatives = spec.get("read_when_any")
    return (holds(spec.get("read_when")) and
            (not alternatives or any(holds(row) for row in alternatives)))


def registry_off_template(template: object,
                          registry: Mapping[str, object] | None = None
                          ) -> object:
    """A template less the entries at their off values.

    A component entry naming the component's :func:`component_off_option`
    and a parameter entry at its :func:`setting_off_value` compose what the
    template without them composes, so a template that gained them (every
    template naming a new component's ``none`` option) keeps its
    identity, and one that names any other option or value moves it.
    """

    if not isinstance(template, Mapping):
        return template
    selected = physics_registry() if registry is None else registry
    kept = dict(template)
    components = template.get("components")
    if isinstance(components, Mapping):
        kept["components"] = {
            component_id: option_id
            for component_id, option_id in components.items()
            if option_id != component_off_option(component_id, selected)}
    parameters = template.get("parameters")
    if isinstance(parameters, Mapping):
        kept["parameters"] = {
            name: value for name, value in parameters.items()
            if not same_setting_value(
                value, setting_off_value(name, selected))}
    return kept


def _selected_option_ids(options: Mapping[str, object] | None,
                         component_id: str,
                         available: Mapping[str, object]) -> tuple[str, ...]:
    if options is None:
        return tuple(available)
    wanted = options.get(component_id)
    return (wanted,) if isinstance(wanted, str) else tuple(wanted or ())


def _unselected_identity_extensions(components, chosen):
    """Additive option facts bind when that option enters the selection.

    Established transition rows remain bound globally as in the prior schema;
    adding an unused option must not invalidate an existing prepared proof.
    """
    excluded = {}
    for component_id, component in components.items():
        if not isinstance(component, Mapping):
            continue
        selected = set(chosen.get(component_id, ()))
        excluded[component_id] = {
            name: option for name, option in component.get("options", {}).items()
            if name not in selected and isinstance(option, Mapping)
            and option.get("physics_identity_extension", {}).get("transitions") is True}
    return excluded


def _transition_identity_scope(transitions, excluded):
    if not isinstance(transitions, Mapping):
        return transitions
    return {name: {**policy, "cross_options": [
        row for row in policy.get("cross_options", ())
        if row.get("parent_option_id") not in excluded.get(policy.get("component_id"), {})
        and row.get("child_option_id") not in excluded.get(policy.get("component_id"), {})]}
        if isinstance(policy, Mapping) and "cross_options" in policy else policy
        for name, policy in transitions.items()}


def _option_identity_scope(option, excluded):
    # MYNN's registry row carries a scheme-indexed F_QS table. Removing
    # only unused additive IDs preserves the prior table and binds every
    # established ID, including deliberate true/false changes to old IDs.
    if not isinstance(option, Mapping):
        return option
    extension = option.get("extensions", {}).get("supplied_moisture_species")
    if not isinstance(extension, Mapping):
        return option
    ids = {value for row in excluded.get("microphysics", {}).values()
           for key, value in row.get("selectors", {}).items() if key == "mp_physics"}
    if not ids:
        return option
    fields = {key: [value for value in values if value not in ids]
              if key in {"flag_qs_true_microphysics_selectors", "flag_qs_false_microphysics_selectors"} else values
              for key, values in extension.items()}
    return {**option, "extensions": {**option["extensions"], "supplied_moisture_species": fields}}


def registry_physics_part_sources(
        registry: Mapping[str, object] | None = None,
        *,
        options: Mapping[str, object] | None = None,
        profile: str | None = None,
) -> dict[str, object]:
    """``part -> registry value`` of the parts a selection runs on, before
    :func:`strip_registry_documentation`; see :func:`registry_physics_parts`.
    """

    selected = physics_registry() if registry is None else registry
    every = options is None
    components = selected.get("components")
    components = components if isinstance(components, Mapping) else {}
    chosen_by_component = {
        component_id: _selected_option_ids(
            options, component_id,
            component.get("options")
            if isinstance(component.get("options"), Mapping) else {})
        for component_id, component in components.items()
        if isinstance(component, Mapping)}
    chosen_options = {
        f"{component_id}.{option_id}"
        for component_id, chosen in chosen_by_component.items()
        for option_id in chosen}
    excluded = {} if every else _unselected_identity_extensions(components, chosen_by_component)
    parts: dict[str, object] = {
        "document": {
            "schema": selected.get("schema"),
            "registry_version": selected.get("registry_version"),
        },
        "transitions": _transition_identity_scope(selected.get("transitions"), excluded),
    }
    parameters = selected.get("parameters")
    if isinstance(parameters, Mapping):
        for knob, spec in parameters.items():
            if isinstance(spec, Mapping) and not parameter_is_implemented(
                    spec):
                continue
            readers = registry_knob_readers(knob, selected)
            if (not every and readers
                    and not readers & chosen_options):
                continue
            parts[f"parameters.{knob}"] = spec
    else:
        parts["parameters"] = parameters
    for component_id, component in components.items():
        if not isinstance(component, Mapping):
            continue
        parts[f"components.{component_id}"] = {
            key: value for key, value in component.items()
            if key != "options"}
        available = component.get("options")
        available = available if isinstance(available, Mapping) else {}
        for option_id in chosen_by_component[component_id]:
            # An option the registry does not carry is a part of its own
            # (null), so a receipt that names one differs by name.
            parts[f"components.{component_id}.options.{option_id}"] = (
                _option_identity_scope(available.get(option_id), excluded))
    templates = selected.get("templates")
    templates = templates if isinstance(templates, Mapping) else {}
    for template_id in (tuple(templates) if every
                        else (() if profile is None else (profile,))):
        canonical = canonical_template_id(template_id, selected)
        # Keep the part key existing preparation receipts bind. Renaming a
        # menu option must not turn unchanged physics into a restart refusal.
        aliases = selected.get("template_aliases", {})
        identity = next((old for old, new in aliases.items() if new == canonical),
                        canonical)
        parts[f"templates.{identity}"] = registry_off_template(
            templates.get(canonical), selected)
    return dict(sorted(parts.items()))


def registry_physics_parts(
        registry: Mapping[str, object] | None = None,
        *,
        options: Mapping[str, object] | None = None,
        profile: str | None = None,
) -> dict[str, str]:
    """``part -> digest`` of the registry parts a selection runs on.

    ``options`` maps a component id to the option ids selected for it
    (one id, or an iterable of ids across a tree's domains); ``None``
    takes every option and every template, which is the whole-registry
    record :data:`REGISTRY_PHYSICS_HISTORY_PATH` keeps.
    """

    return {
        name: canonical_sha256(strip_registry_documentation(value))
        for name, value in registry_physics_part_sources(
            registry, options=options, profile=profile).items()
    }


def registry_physics_receipt(
        registry: Mapping[str, object] | None = None,
        *,
        options: Mapping[str, object],
        profile: str | None = None,
) -> dict[str, object]:
    """The registry physics a selection receipt records, part by part."""

    return {
        "schema": REGISTRY_PHYSICS_IDENTITY_SCHEMA,
        "parts": registry_physics_parts(
            registry, options=options, profile=profile),
    }


def registry_physics_sha256(
        registry: Mapping[str, object] | None = None) -> str:
    """One digest of the WHOLE registry's physics (every part).

    Documentation, citations, labels, maturity and admission rules leave
    it unchanged; any other option, parameter, transition or template
    change moves it.  It keys
    :data:`REGISTRY_PHYSICS_HISTORY_PATH`; a selection binds its parts.
    """

    return canonical_sha256({
        "schema": REGISTRY_PHYSICS_IDENTITY_SCHEMA,
        "parts": registry_physics_parts(registry),
    })


#: The registry documents a selection receipt bound BEFORE receipts
#: carried ``registry_physics``, each with its physics parts.  Such a
#: receipt records only the document digest it was prepared under
#: (``registry_sha256``); this record is how it resolves.  It holds the
#: 2.8 line's documents from the 2.8.0 release on and is written by
#: ``tools/registry_physics_history.py`` from the committed documents, so
#: every row is a fact about a document git holds.  A document not in it
#: is refused by name, as one whose physics this build cannot establish.
REGISTRY_PHYSICS_HISTORY_PATH = Path(__file__).with_name(
    "physics_registry_history.json")
REGISTRY_PHYSICS_HISTORY_SCHEMA = "gpuwm-physics-registry-history-v1"

_registry_physics_history: dict[str, object] | None = None


def registry_physics_history() -> dict[str, object]:
    """The legacy record: ``documents`` (digest -> physics) and ``physics``
    (physics digest -> parts)."""

    global _registry_physics_history
    if _registry_physics_history is None:
        from woof import packed_data
        # (WOOF 1.1.6) a wheel carries it as physics_registry_history.json.xz
        # (woof.packed_data, PyPI's per-file limit); same bytes either way.
        value = json.loads(
            packed_data.read_text(REGISTRY_PHYSICS_HISTORY_PATH),
            parse_constant=_reject_json_constant)
        if (not isinstance(value, dict)
                or value.get("schema") != REGISTRY_PHYSICS_HISTORY_SCHEMA
                or value.get("identity_schema")
                != REGISTRY_PHYSICS_IDENTITY_SCHEMA
                or not isinstance(value.get("documents"), dict)
                or not isinstance(value.get("physics"), dict)):
            raise RuntimeError(
                f"{REGISTRY_PHYSICS_HISTORY_PATH.name} is not a "
                f"{REGISTRY_PHYSICS_HISTORY_SCHEMA} record for "
                f"{REGISTRY_PHYSICS_IDENTITY_SCHEMA}")
        _registry_physics_history = value
    return deepcopy(_registry_physics_history)


def recorded_registry_physics_parts(
        receipt: Mapping[str, object]) -> dict[str, str] | None:
    """The physics parts a selection receipt was written under, or None.

    A receipt that carries ``registry_physics`` names its parts.  One
    written before that field existed, or in another identity schema,
    resolves through the document digest it was prepared under to every
    part that document had in the history record; ``None`` means that
    document is not in it.
    """

    recorded = receipt.get("registry_physics")
    if isinstance(recorded, Mapping):
        parts = recorded.get("parts")
        if (recorded.get("schema") == REGISTRY_PHYSICS_IDENTITY_SCHEMA
                and isinstance(parts, Mapping)):
            return dict(parts)
    # Parts in another identity schema are not comparable with this one's;
    # the document digest every receipt records still is, through the
    # history.
    document = receipt.get("registry_sha256")
    if not isinstance(document, str):
        return None
    history = registry_physics_history()
    row = history["documents"].get(document)
    physics = row.get("physics_sha256") if isinstance(row, Mapping) else None
    parts = history["physics"].get(physics) if isinstance(
        physics, str) else None
    return dict(parts) if isinstance(parts, Mapping) else None


# ----------------------------------------------------------- consumer rows
#: The per-option object under which the registry publishes what every
#: CONSUMER of a scheme needs -- the checkpoint identity, the vertical
#: bound, the nest-edge closure, the radar operator's floor and route, the
#: moment structure, the offline-child admission, the ring-guard field set,
#: the reflectivity inputs, the stock-WRF export inventory.  Generated by
#: ``tools/build_registry.py``; never typed into the JSON by hand.
#:
#: WHY IT EXISTS.  On 2026-09-10 a Milbrandt-Yau (mp_physics=9) forecast the
#: loader accepted, the dispatcher ran and the registry called implemented
#: integrated for 59 minutes and died writing its first hourly checkpoint,
#: because one of forty-four hand-kept scheme lists in the tree had never
#: learned the scheme.  Plan review checked only the registry; the consumer
#: lists were consulted for the first time inside the run.  These rows are
#: the single source those lists are now derived from or asserted equal
#: to, and :func:`consumer_row_gaps` is the plan-review question "does
#: every consumer this run will reach have its row" -- asked before step 0.
CONSUMER_ROWS_KEY = "consumers"

#: Set to ``"1"`` by ``tools/build_registry.py`` while it REBUILDS the
#: registry.  The import-time agreement checks consumers run through
#: :func:`require_registry_agreement` compare a module's table against the
#: registry on disk; while the registry is being regenerated from those very
#: modules the disk copy is the stale one, so the check would refuse the
#: rebuild that fixes it.  Nothing else sets this, and the byte-equality
#: test in ``tests/test_build_registry.py`` is what catches a stale registry
#: the rebuild did not follow.
REGISTRY_REBUILD_ENV = "WOOF_PHYSICS_REGISTRY_REBUILD"

#: The one key of a route's ``expert_template_ids`` that names no source:
#: templates declared under it are offered for every source the route
#: lists in ``source_ids``.  A per-source key stays legal beside it for a
#: declaration that really is one source's own.  Every reader resolves a
#: source through :func:`expert_template_ids_for_source` so the two shapes
#: cannot drift apart.
EXPERT_TEMPLATES_ANY_SOURCE = "*"


def expert_template_ids_for_source(route: Mapping[str, object] | None,
                                   source_id: object) -> list[str]:
    """Expert templates ``route`` declares for ``source_id``.

    The source's own list first, then the route-wide list under
    :data:`EXPERT_TEMPLATES_ANY_SOURCE`, each id once, in declaration
    order.  The route-wide list does not reach a source whose own
    ``source_template_ids`` entry is an empty list: such a source (a
    mapped configuration, for one) runs from no template at all, and
    offering it an expert template would offer what it cannot take.
    Anything that is not a list of strings contributes nothing.
    """
    if not isinstance(route, Mapping):
        return []
    expert = route.get("expert_template_ids", {})
    if not isinstance(expert, Mapping):
        return []
    normal = route.get("source_template_ids", {})
    offers_nothing = (
        isinstance(normal, Mapping) and isinstance(source_id, str)
        and isinstance(normal.get(source_id), list)
        and not normal.get(source_id))
    keys = (source_id,) if offers_nothing else (
        source_id, EXPERT_TEMPLATES_ANY_SOURCE)
    resolved: list[str] = []
    for key in keys:
        declared = expert.get(key, []) if isinstance(key, str) else []
        if isinstance(declared, list):
            for value in declared:
                if isinstance(value, str) and value not in resolved:
                    resolved.append(value)
    return resolved

#: Per component: the consumer-row keys every IMPLEMENTED option carrying
#: selectors must publish, and the consumer that reads each key -- which is
#: the breakage a missing row would otherwise become.  The builder refuses
#: to write a registry that violates this; plan review refuses to launch a
#: configuration whose resolved option violates it (a hand-edited JSON, a
#: consumer added ahead of its rows).  Turbulence carries no identity row
#: by design: ``configuration_sha256`` binds km_opt and its constants, and
#: a row would be a sixth turbulence authority; the operators'
#: implementation is the one string
#: ``woof.checkpoint_identity.DYCORE_MIXING_ALGORITHM_IDENTITY``.
CONSUMER_ROW_CONTRACT: dict[str, dict[str, str]] = {
    "microphysics": {
        "restart_algorithm_identity": (
            "the checkpoint writer names every scheme it serialises "
            "(woof/checkpoint_identity.py); without a row the FIRST "
            "restart interval raises and the forecast to that point is lost"),
        "vertical_level_bounds": (
            "the preparation-time vertical preflight "
            "(woof/physics_compat.py) reads the level window here; without "
            "a row the first microphysics call is the first check"),
        "nest_transition": (
            "the nest-edge resolver (woof/core/microphysics_transition.py) "
            "decides at tree load whether a mixed parent/child edge has a "
            "closure"),
        "radar_da": (
            "the radar operator's clear-air floor and H(x) route "
            "(woof/da/obsop.py), consulted at the first analysis"),
        "moments": (
            "the LETKF moment policy (woof/da/moments.py) derives the "
            "analysis state vector from this row at the first analysis"),
        "offline_child": (
            "the offline downscale lane (woof/offline_child.py) admits or "
            "refuses a parent scheme at scan time from this row"),
        "lateral_forcing_dataset": (
            "the dataset a domain of this scheme with EXTERNAL lateral "
            "boundaries (specified) needs before it can be forced "
            "correctly; plan review resolves it here through the row's own "
            "named resolver, which is the same code object the run door "
            "raises from, so the two authorities cannot decide it "
            "differently"),
        "ring_guard": (
            "the specified-zone ring guard (woof/core/physics_inventory.py, "
            "woof/core/microphysics.py) prices and captures exactly these "
            "fields at run start and on every microphysics step"),
        "reflectivity_input_species": (
            "the reflectivity operator and the streamed composite "
            "(woof/core/refl.py) read these state species"),
        "stock_wrf_export": (
            "the stock-WRF export inventory (woof/wrf_physics_inventory.py, "
            "woof/wrf_direct.py) writes exactly these wrfinput members"),
        "cloud_optics": (
            "the RTE+RRTMGP adapter (woof/core/rrtmgp.py) resolves the "
            "scheme's cloud-optics coupling and the legacy RRTMG adapter "
            "(woof/core/rrtmg_legacy.py) its use_mp_re declaration at the "
            "first radiation call"),
    },
    "cumulus": {
        "restart_algorithm_identity": (
            "the checkpoint writer names the cumulus scheme at the first "
            "restart interval"),
        "stock_callable_class": (
            "the checkpoint writer recognises the stock adapter class by this "
            "name; a scheme without one is routed down the custom-callable "
            "path at the first restart interval"),
        "vertical_level_bounds": (
            "the preparation-time vertical preflight reads the level window "
            "here"),
    },
    "pbl": {
        "scalar_diffusion_exchange": (
            "scalar_pblmix admission, the post-PBL diff4d coupling and VRAM "
            "admission read the EXCH_H producer, its levels and the no-op "
            "reason here"),
        "restart_algorithm_identity": (
            "the checkpoint writer names the PBL scheme at the first restart "
            "interval"),
        "vertical_level_bounds": (
            "the preparation-time vertical preflight reads the level window "
            "here"),
    },
    "surface_layer": {
        "restart_algorithm_identity": (
            "the checkpoint writer names the surface-layer scheme at the "
            "first restart interval"),
    },
    "urban": {
        "restart_algorithm_identity": (
            "the checkpoint writer binds the urban model whose arrays ride "
            "the surface inventory (physics_setup_identity), so a resume "
            "across urban models is refused"),
        "vertical_level_bounds": (
            "no urban model bounds the column: the UCM is a surface model "
            "and BEP/BEM map their own urban grid onto whatever levels the "
            "PBL admits"),
    },
    "land_surface": {
        "restart_algorithm_identity": (
            "the checkpoint writer names the land-surface scheme at the "
            "first restart interval"),
        "restart_parameter_bundle": (
            "the checkpoint writer binds the packed parameter bundle's bytes "
            "through this row; a scheme without one raises the same "
            "RestartManifestError at the same point in the run"),
        "reads_glw": (
            "the radiation-off/longwave-off guards (woof/physics_compat.py) "
            "decide from this whether an absent longwave stream is a wrong "
            "forecast or an unused buffer"),
    },
    "radiation": {
        "restart_algorithm_identity": (
            "the checkpoint writer names both spectra and their "
            "above-atmosphere policies at the first restart interval"),
        "stock_callable_class": (
            "the checkpoint writer recognises the stock radiation adapter by "
            "this name; a selection without one is routed down the "
            "custom-callable path at the first restart interval"),
    },
    "turbulence": {
        "restart_identity_binding": (
            "records that km_opt is bound by configuration_sha256, has no "
            "per-option identity table by design, and that the mixing "
            "operators' implementation is bound by the one "
            "DYCORE_MIXING_ALGORITHM_IDENTITY string"),
    },
}


def registry_view() -> Mapping[str, object]:
    """The loaded registry itself, for READ-ONLY derivation at import time.

    :func:`physics_registry` deep-copies a 300 KB document on every call,
    which is right for a caller that may mutate it and wrong for the dozen
    consumer modules that derive one table each at import.  Callers of this
    function must not mutate the result.
    """

    return _REGISTRY


def component_options(component_id: str, *,
                      implemented_only: bool = True) -> dict[str, Mapping]:
    """The registered options of one component, by option id."""

    components = _REGISTRY.get("components")
    component = components.get(component_id) if isinstance(components, dict) else None
    if not isinstance(component, dict):
        raise KeyError(f"physics registry has no component {component_id!r}")
    options = component.get("options", {})
    return {
        option_id: option
        for option_id, option in options.items()
        if isinstance(option, dict)
        and (not implemented_only or option.get("implemented") is True)
    }



def column_microphysics_binding(selector: int):
    """Optional declarative bindings for the shared column-kernel launcher.

    Older production routes keep their existing dispatch. New column ports
    declare allocations, argument groups and history names in their option
    row, rather than adding another scheme-specific adapter file.
    """
    for option in component_options('microphysics').values():
        if int(option.get('selectors', {}).get('mp_physics', -1)) == int(selector):
            return option.get('extensions', {}).get('column_kernel')
    return None

def component_selector_key(component_id: str) -> str:
    """The ONE selector key of a single-selector component.

    Radiation selects on a (lw, sw) pair and has no single key; asking for
    one is a caller error rather than a silent first element.
    """

    component = _REGISTRY["components"][component_id]
    keys = list(component.get("selector_keys", []))
    if len(keys) != 1:
        raise ValueError(
            f"component {component_id!r} selects on {keys}, not on one key")
    return keys[0]


def implemented_selector_values(component_id: str) -> dict[int, str]:
    """``{selector value: option id}`` over a component's implemented options.

    The inventory every consumer table restates: a hand-kept list of scheme
    ids is drift unless it equals these keys (or names, by defect id, each
    key it deliberately lacks -- see :func:`require_registry_agreement`).
    """

    key = component_selector_key(component_id)
    rows: dict[int, str] = {}
    for option_id, option in component_options(component_id).items():
        selectors = option.get("selectors", {})
        if isinstance(selectors, dict) and key in selectors:
            rows[int(selectors[key])] = option_id
    return rows


def option_consumer_row(component_id: str, option_id: str, key: str):
    """One consumer row of one option; ``KeyError`` names the gap."""

    option = component_options(component_id, implemented_only=False)[option_id]
    rows = option.get(CONSUMER_ROWS_KEY)
    if not isinstance(rows, dict) or key not in rows:
        raise KeyError(
            f"components.{component_id}.options.{option_id} publishes no "
            f"{CONSUMER_ROWS_KEY}.{key} row; "
            + CONSUMER_ROW_CONTRACT.get(component_id, {}).get(
                key, "a consumer reads it"))
    return rows[key]


def consumer_rows_by_selector(component_id: str, key: str) -> dict[int, object]:
    """``{selector value: consumers[key]}`` over implemented options.

    Options that publish no such row are omitted rather than defaulted, so
    a consumer deriving its table from this sees exactly the rows the
    registry carries and :func:`consumer_row_gaps` reports the rest.
    """

    result: dict[int, object] = {}
    for value, option_id in implemented_selector_values(component_id).items():
        option = component_options(component_id)[option_id]
        rows = option.get(CONSUMER_ROWS_KEY)
        if isinstance(rows, dict) and key in rows:
            result[value] = rows[key]
    return result


def consumer_row_for_selector(component_id: str, key: str, value) -> object | None:
    """``consumers[key]`` of the implemented option selected by ``value``.

    ``None`` when no implemented option carries that selector value, so a
    door that is not a RunConfig (the DA cycle configuration, which
    validate_run_config never sees) can ask the registry the same
    question at ITS plan review instead of at its first analysis.
    """

    return consumer_rows_by_selector(component_id, key).get(int(value))


def template_ids_with_components(**components: str) -> tuple[str, ...]:
    """Registered template ids whose composition names every given option.

    ``components`` maps component id -> option id.  A consumer that must
    cite a template it deliberately omits names it THROUGH its
    composition rather than by its id literal, so a template whose id
    carries a source or case token can be cited from a module where such
    tokens may not appear, and a renamed template keeps its citation.
    """

    matched: list[str] = []
    for template_id, record in sorted(_REGISTRY.get("templates", {}).items()):
        selected = record.get("components", {}) if isinstance(record, dict) else {}
        if all(selected.get(component_id) == option_id
               for component_id, option_id in components.items()):
            matched.append(template_id)
    return tuple(matched)


def require_registry_agreement(consumer: str, component_id: str,
                               observed, *,
                               cited_absences: Mapping[int, str] | None = None,
                               cited_extras: Mapping[int, str] | None = None,
                               ) -> None:
    """Refuse, AT IMPORT, a scheme table that disagrees with the registry.

    ``observed`` is the consumer's own key set.  It must equal the
    registry's implemented selector values for ``component_id``, except for
    keys the consumer deliberately lacks (``cited_absences``: id -> the
    defect or breakage that keeps it out, so the retirement sweep is a
    grep) and keys it deliberately carries beyond the registry
    (``cited_extras``, same shape).  A cited id that is in fact present, or
    an uncited difference in either direction, is the drift this exists to
    stop, and it is reported as a ``RuntimeError`` from the consumer's own
    import rather than from a forecast an hour in.

    Skipped only while ``tools/build_registry.py`` regenerates the registry
    (:data:`REGISTRY_REBUILD_ENV`), because the disk copy is then the stale
    side by construction.
    """

    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        return
    registry_keys = set(implemented_selector_values(component_id))
    observed_keys = {int(value) for value in observed}
    absences = dict(cited_absences or {})
    extras = dict(cited_extras or {})
    problems: list[str] = []
    missing = registry_keys - observed_keys
    surplus = observed_keys - registry_keys
    for value in sorted(missing - set(absences)):
        problems.append(
            f"the registry implements {component_id} selector {value} and "
            f"{consumer} has no row for it")
    for value in sorted(set(absences) - missing):
        problems.append(
            f"{consumer} cites {value} as deliberately absent ({absences[value]}) "
            "but the row is present or the registry does not implement it; "
            "retire the citation")
    for value in sorted(surplus - set(extras)):
        problems.append(
            f"{consumer} carries {component_id} selector {value}, which the "
            "registry does not implement")
    for value in sorted(set(extras) - surplus):
        problems.append(
            f"{consumer} cites {value} as a deliberate extra ({extras[value]}) "
            "but it is not one; retire the citation")
    if problems:
        raise RuntimeError(
            f"{consumer} disagrees with woof/physics_registry_v2.json about "
            f"{component_id}: " + "; ".join(problems)
            + ".  A scheme is rows in the registry, not a code path: give the "
            "option its row (tools/build_registry.py) or cite the defect that "
            "keeps it out.")


def require_consumer_rows_agreement(consumer: str, component_id: str,
                                    key: str, observed: Mapping[int, object],
                                    *, project=None,
                                    cited_absences: Mapping[int, str] | None = None,
                                    ) -> None:
    """Refuse, AT IMPORT, a consumer table whose VALUES left the registry's.

    The registry's ``consumers.<key>`` rows for ``component_id`` were
    generated from the module that owns the fact; this holds the module to
    the generated copy so neither can move without the other.  ``observed``
    is ``{selector value: the module's value}``; ``project`` reduces a
    registry row to the same shape (identity by default).  A selector the
    registry implements and the module lacks must be cited in
    ``cited_absences`` (id -> defect or breakage), exactly as in
    :func:`require_registry_agreement`.  Skipped during a registry rebuild.
    """

    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        return
    rows = consumer_rows_by_selector(component_id, key)
    absences = dict(cited_absences or {})
    problems: list[str] = []
    for value in sorted(rows):
        expected = rows[value] if project is None else project(rows[value])
        if value in observed:
            if value in absences:
                problems.append(
                    f"selector {value} is cited as deliberately absent "
                    f"({absences[value]}) but {consumer} carries it; retire "
                    "the citation")
            elif observed[value] != expected:
                problems.append(
                    f"selector {value}: {consumer} says {observed[value]!r}, "
                    f"the registry row says {expected!r}")
        elif value not in absences:
            problems.append(
                f"the registry implements selector {value} and {consumer} "
                "has no row for it")
    for value in sorted(set(observed) - set(rows)):
        problems.append(
            f"{consumer} carries selector {value}, for which the registry "
            f"publishes no {CONSUMER_ROWS_KEY}.{key} row")
    for value in sorted(set(absences) - set(rows)):
        problems.append(
            f"{consumer} cites {value} as deliberately absent "
            f"({absences[value]}) but the registry has no such implemented "
            "option; retire the citation")
    if problems:
        raise RuntimeError(
            f"{consumer} disagrees with woof/physics_registry_v2.json "
            f"({component_id} {CONSUMER_ROWS_KEY}.{key}): "
            + "; ".join(problems)
            + ".  Change the owning module and regenerate the registry with "
            "tools/build_registry.py, or cite the defect that keeps a row "
            "out.")


def require_template_menu_agreement(consumer: str, menu,
                                    *, cited_absences: Mapping[str, str] | None = None,
                                    ) -> None:
    """Refuse, AT IMPORT, a profile menu that drifted from the templates.

    A hand-kept list of template ids is a scheme table too: an implemented
    composition with no menu row has no front door.  ``menu`` must equal
    the registry's template ids, each deliberately omitted template named
    in ``cited_absences`` (template id -> the reason it has no row), and a
    menu id that names no template is refused outright.

    Skipped while ``tools/build_registry.py`` regenerates the registry
    (:data:`REGISTRY_REBUILD_ENV`), exactly as the two agreement checks
    above are, and for exactly their reason: the builder reaches this
    module through the loader it derives the route declarations from, so
    the disk copy it is about to replace is the stale side by
    construction.  Without the guard, a registry that needs a rebuild
    refuses the rebuild that would fix it.
    """

    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        return
    templates = set(_REGISTRY.get("templates", {}))
    observed = set(menu)
    absences = dict(cited_absences or {})
    problems: list[str] = []
    for template_id in sorted(templates - observed - set(absences)):
        problems.append(
            f"template {template_id!r} is registered and {consumer} offers "
            "no row for it")
    for template_id in sorted(set(absences) - (templates - observed)):
        problems.append(
            f"{consumer} cites {template_id!r} as deliberately absent "
            f"({absences[template_id]}) but it is offered or not registered; "
            "retire the citation")
    for template_id in sorted(observed - templates):
        problems.append(
            f"{consumer} offers {template_id!r}, which names no registered "
            "template")
    if len(observed) != len(list(menu)):
        problems.append(f"{consumer} lists a template twice")
    if problems:
        raise RuntimeError(
            f"{consumer} disagrees with woof/physics_registry_v2.json "
            "templates: " + "; ".join(problems)
            + ".  Add the menu row, or cite the reason the template has none.")


def _selection(settings, key: str):
    """A selector value from a Mapping or an attribute-carrying object."""

    if isinstance(settings, Mapping):
        return settings.get(key)
    return getattr(settings, key, None)


def _option_for_selectors(component: Mapping, selected: Mapping) -> tuple[str, Mapping] | None:
    options = component.get("options", {})
    for option_id, option in options.items():
        if not isinstance(option, dict):
            continue
        selectors = option.get("selectors", {})
        if (isinstance(selectors, dict) and selectors
                and set(selectors) == set(selected)
                and all(_same_value(selected[key], selectors[key])
                        for key in selectors)):
            return option_id, option
    return None


def consumer_row_gaps_for_option(component_id: str, option_id: str,
                                 option: Mapping, settings) -> list[str]:
    """Every consumer row this option lacks for the run ``settings`` describe.

    Presence first: each key of :data:`CONSUMER_ROW_CONTRACT` for the
    component.  Then the value-level questions a run actually asks: an
    active cumulus or radiation selection needs a stock class name, an
    active land surface a parameter bundle, an active microphysics scheme a
    level window, a ring-guard field set and a moment structure, and a 4/4
    radiation selection a class for the ``ra_rrtmg_variant`` it runs.
    """

    contract = CONSUMER_ROW_CONTRACT.get(component_id)
    if contract is None or option.get("implemented") is not True:
        return []
    selectors = option.get("selectors", {})
    rows = option.get(CONSUMER_ROWS_KEY)
    label = f"{component_id} option {option_id!r} ({selectors})"
    gaps: list[str] = []
    if not isinstance(rows, dict):
        return [f"{label} publishes no {CONSUMER_ROWS_KEY} block at all; "
                + "; ".join(contract.values())]
    for key, consumer in contract.items():
        if key not in rows:
            gaps.append(f"{label} has no {CONSUMER_ROWS_KEY}.{key} row: {consumer}")
    if gaps:
        return gaps
    is_off = all(int(value) == 0 for value in selectors.values()
                 if isinstance(value, int) and not isinstance(value, bool))
    if component_id == "microphysics" and not is_off:
        for key in ("vertical_level_bounds", "ring_guard", "moments",
                    "restart_algorithm_identity", "cloud_optics"):
            if rows.get(key) is None:
                gaps.append(f"{label} carries a null {key} row: "
                            f"{contract[key]}")
    if component_id in ("cumulus", "pbl") and not is_off:
        if rows.get("vertical_level_bounds") is None:
            gaps.append(f"{label} carries a null vertical_level_bounds row: "
                        f"{contract['vertical_level_bounds']}")
    if component_id == "cumulus" and not is_off:
        if not rows.get("stock_callable_class"):
            gaps.append(f"{label} carries no stock_callable_class: "
                        f"{contract['stock_callable_class']}")
    if component_id == "land_surface" and not is_off:
        if rows.get("restart_parameter_bundle") is None:
            gaps.append(f"{label} carries a null restart_parameter_bundle: "
                        f"{contract['restart_parameter_bundle']}")
    if component_id == "radiation" and not is_off:
        stock = rows.get("stock_callable_class")
        if isinstance(stock, dict):
            variant = _selection(settings, "ra_rrtmg_variant")
            if variant is None:
                spec = _REGISTRY.get("parameters", {}).get("ra_rrtmg_variant", {})
                variant = spec.get("default") if isinstance(spec, dict) else None
            if variant not in stock:
                gaps.append(
                    f"{label} names no stock_callable_class for "
                    f"ra_rrtmg_variant={variant!r} (known: {sorted(stock)}): "
                    f"{contract['stock_callable_class']}")
        elif not stock:
            gaps.append(f"{label} carries no stock_callable_class: "
                        f"{contract['stock_callable_class']}")
    return gaps


def consumer_row_gaps(settings) -> list[str]:
    """PLAN REVIEW: the consumer rows the selected physics tuple lacks.

    ``settings`` is a RunConfig or any Mapping/object carrying the
    component selector keys.  Each component the settings select is
    resolved to its registry option through the selectors alone; a
    component whose selectors the settings never mention is not selected
    and is skipped, and a radiation pair with no registered preset (an
    independently composed lw/sw pair) resolves to no option and is
    skipped too -- the composition declares its own restart identity.
    Unknown or unimplemented selector values are somebody else's refusal
    (the loader's, made earlier and by name) and are not repeated here.
    Empty is the answer for every configuration every consumer can serve.
    """

    gaps: list[str] = []
    components = _REGISTRY.get("components", {})
    for component_id, component in sorted(components.items()):
        if not isinstance(component, dict):
            continue
        keys = [key for key in component.get("selector_keys", [])
                if isinstance(key, str)]
        if not keys:
            continue
        selected = {key: _selection(settings, key) for key in keys}
        if any(value is None for value in selected.values()):
            continue
        found = _option_for_selectors(component, selected)
        if found is None:
            continue
        option_id, option = found
        gaps.extend(consumer_row_gaps_for_option(
            component_id, option_id, option, settings))
    return gaps


def require_consumer_rows(settings) -> None:
    """Refuse, AT PLAN REVIEW, a run some consumer could not serve.

    THE CONCRETE BREAKAGE: a scheme every front door admits whose row is
    missing from one downstream table dies where that table is first read
    -- the first checkpoint, the first analysis, the first nest edge, the
    first reflectivity frame, the first microphysics call -- after the run
    has spent its preparation and part of its integration.  This asks every
    consumer's question while the answer is still free, from
    ``validate_run_config``, with no flag to set.  After the registry
    carries every row this is silent; it exists so the next drift costs a
    launch and never a run.
    """

    gaps = consumer_row_gaps(settings)
    if not gaps:
        return
    raise ValueError(
        "this configuration selects a scheme some consumer cannot serve: "
        + "; ".join(gaps)
        + ".  The way out is a row, not a flag: give the option its "
        f"{CONSUMER_ROWS_KEY} row in tools/build_registry.py and regenerate "
        "woof/physics_registry_v2.json, or select a scheme that has one.")


def stock_callable_class(component_id: str, selectors: Mapping[str, object],
                         *, variant: str | None = None) -> str | None:
    """The stock adapter class the checkpoint writer expects, or ``None``.

    ``None`` means the registry knows the selection and it has no stock
    class (an ``off`` option); an UNKNOWN selection raises ``KeyError`` so
    the caller can say "no stock-class row" rather than "custom callable".
    For a 4/4 radiation selection ``variant`` picks the adapter.
    """

    component = _REGISTRY["components"][component_id]
    found = _option_for_selectors(component, dict(selectors))
    if found is None:
        raise KeyError(
            f"no registered {component_id} option matches {dict(selectors)}")
    option_id, option = found
    stock = option_consumer_row(component_id, option_id, "stock_callable_class")
    if isinstance(stock, dict):
        if variant is None:
            spec = _REGISTRY.get("parameters", {}).get("ra_rrtmg_variant", {})
            variant = spec.get("default") if isinstance(spec, dict) else None
        if variant not in stock:
            raise KeyError(
                f"{component_id} option {option_id!r} has no stock-class row "
                f"for ra_rrtmg_variant={variant!r}; known: {sorted(stock)}")
        return stock[variant]
    return stock


def _templates_carrying(templates: Mapping[str, object], candidates,
                        component_id: str, option_id: str) -> list[str]:
    """Registered template ids among ``candidates`` whose ``component_id``
    is ``option_id`` -- the way to an option a route will not let a plan
    override into."""

    carrying = []
    for template_id in sorted(candidates):
        template = templates.get(template_id)
        if not isinstance(template, dict):
            continue
        components = template.get("components", {})
        if isinstance(components, dict) and components.get(component_id) == option_id:
            carrying.append(template_id)
    return carrying


def _acknowledged(carrying, expert_templates, acknowledgement) -> str:
    """Name an expert template's advisory acknowledgement without requiring it."""
    if not acknowledgement or not any(
            template_id in expert_templates for template_id in carrying):
        return ""
    return (f"; {acknowledgement!r} acknowledges the expert-template advisory "
            "and is optional")


def _deferred_parameter_clause(runner_id, deferred, route_parameter_keys) -> str:
    """What a route says about a per-domain setting it does not take.

    Three cases, and none of them is the bare "runner route does not
    accept this per-domain setting" that used to be the whole message.
    A route that DEFERS the knob published the component it belongs to
    and the way to the value in ``deferred_parameter_keys``, and that
    sentence is printed verbatim.  A route that takes some settings per
    domain names the declaration that lists them.  A route that takes
    none says so, because pointing a reader at an empty list is a way out
    that goes nowhere.
    """

    if isinstance(deferred, str) and deferred:
        return f": {deferred}"
    if not isinstance(runner_id, str):
        return ("; this plan names no runner route, so no per-domain "
                "setting is accepted")
    if not route_parameter_keys:
        return (f"; runner_routes.{runner_id}.allowed_parameter_keys is "
                "empty, so this route takes no per-domain setting at all "
                "and the value belongs to the configuration its runner "
                "replays rather than to one domain")
    return (f"; the settings it does take per domain are "
            f"runner_routes.{runner_id}.allowed_parameter_keys")


def component_override_declaration(route: Mapping[str, object]) -> str:
    """Describe declared override paths without inferring benchmark semantics."""

    raw_free = route.get("allowed_component_overrides", [])
    free = (sorted(value for value in raw_free if isinstance(value, str))
            if isinstance(raw_free, list) else [])
    raw_options = route.get("allowed_component_options", {})
    scoped = sorted(
        key for key, values in raw_options.items()
        if isinstance(key, str) and isinstance(values, list) and values
    ) if isinstance(raw_options, Mapping) else []
    return (
        f"this route declares whole-component override axes {free!r} and "
        f"explicit option lists for {scoped!r}; all selections remain subject "
        "to their option constraints. A different selection has no declared "
        "plan-level override path. Choose an admitted option or a reachable "
        "base template carrying it")


def _route_override_reason(routes, runner_id, component_id) -> str:
    """What THIS route says an override of THIS component would break.

    ``tools/build_registry.py`` writes the breakage twice over: once per
    component in ``component_override_refusal_reasons``, for a route whose
    own lists exclude an implemented option, and once per route in
    ``component_override_refusal_reason`` for a route where the refusal is
    the same whatever component is named.  Both were written and read by
    nothing, so every route printed one hard-coded clause -- "runs its
    registered templates unchanged" -- which after the 2026-07-31 ruling
    described only the benchmark route, while the single-domain route was
    refusing land_surface noah-mp with it seconds after admitting a
    cumulus, microphysics or turbulence override on the same plan.

    The per-component key is read first because a refusal is delivered
    about ONE component: a route-wide reason covering two exclusions told
    a user who named land_surface what is wrong with an analytic radiation
    scheme, which is a breakage that did not fire.
    """

    route = routes.get(runner_id) if isinstance(routes, Mapping) else None
    if not isinstance(route, dict):
        return ""
    per_component = route.get("component_override_refusal_reasons")
    if isinstance(per_component, Mapping):
        reason = per_component.get(component_id)
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    reason = route.get("component_override_refusal_reason")
    return reason.strip() if isinstance(reason, str) and reason.strip() else ""


def _override_route_refusal(runner_id, routes, source_id, templates, reachable,
                            template_id, template_components, component_id,
                            option_id, route_component_overrides,
                            route_component_options, expert_templates=(),
                            acknowledgement=None) -> str:
    """Why an experiment-per-domain route turns a component override away.

    The route declares which components may vary per domain and which
    options the others may take; an option outside both is refused by
    declaration, not by any runtime limit, and the sentence says so and
    names the templates that reach the option on this route and source.
    """

    template_option = (template_components.get(component_id)
                       if isinstance(template_components, dict) else None)
    free = sorted(route_component_overrides)
    admitted = sorted(route_component_options.get(component_id, ()))
    carrying = _templates_carrying(templates, reachable, component_id, option_id)
    declared = (
        f"runner {runner_id!r} lets "
        + (", ".join(free) if free else "no component")
        + " vary per domain and admits " + component_id + " only "
        + (f"from {admitted!r} or " if admitted else "")
        + "as the base template carries it")
    if carrying:
        way_out = (
            f"Start from a registered template that carries {component_id} "
            f"{option_id!r} on this route and source: {carrying!r}"
            + _acknowledged(carrying, expert_templates, acknowledgement) + ".")
    else:
        way_out = (
            f"No registered template reachable on this route for source "
            f"{source_id!r} carries {component_id} {option_id!r}, so the "
            "option is unreachable here by route declaration "
            "(runner_routes in the physics registry), not by a runtime "
            "limit; widening the route's allowed_component_options or "
            "registering a template that carries it is the way to it.")
    reason = _route_override_reason(routes, runner_id, component_id)
    return (
        f"option {option_id!r} of {component_id} cannot replace the base "
        f"template {template_id!r}'s {template_option!r} on this domain: "
        f"{declared}, so this plan would run a {component_id} the route "
        f"has not declared runnable per domain."
        + (f" The route states why: {reason.rstrip('.')}." if reason else "")
        + f" {way_out}")


def _fixed_template_override_refusal(runner_id, routes, templates, reachable,
                                     template_id, template_components,
                                     component_id, option_id,
                                     expert_templates=(),
                                     acknowledgement=None) -> str:
    """Why THIS fixed-template route turns this component override away.

    The reason is the route's own declared reason for THIS component
    (``component_override_refusal_reasons``, falling back to the
    route-wide ``component_override_refusal_reason``), because the two
    fixed-template routes stopped refusing the same thing: the benchmark
    route replays
    an immutable template, and the prepared single-domain route declares
    the tree route's component overrides (cumulus, microphysics and
    turbulence vary freely; land surface, PBL, radiation and surface
    layer resolve to any option it lists) and excludes exactly two
    implemented options, each with a way out.  Printing "runs its
    registered templates unchanged" for both told a user the route is
    immutable moments after the same validator admitted four other
    overrides on it, and pointed them away from the real reason.  A route
    that declares no reason gets a description of its declared override
    paths, not an inferred benchmark or comparison contract.
    """

    template_option = (template_components.get(component_id)
                       if isinstance(template_components, dict) else None)
    carrying = _templates_carrying(templates, reachable, component_id, option_id)
    experiment_routes = sorted(
        route_id for route_id, route in routes.items()
        if isinstance(route, dict)
        and route.get("mode") == "experiment-per-domain"
        and route.get("implemented") is True)
    ways = []
    if carrying:
        ways.append(
            f"select a registered template that carries {component_id} "
            f"{option_id!r} on this route: {carrying!r}"
            + _acknowledged(carrying, expert_templates, acknowledgement))
    if experiment_routes:
        ways.append(
            "run the composition on an experiment-per-domain runner "
            f"({', '.join(experiment_routes)}), which admits per-domain "
            "component overrides")
    if not ways:
        ways.append(
            "no registered template carries it and no implemented runner "
            "admits per-domain overrides, so the option is unreachable by "
            "route declaration (runner_routes in the physics registry)")
    reason = _route_override_reason(routes, runner_id, component_id) or (
        component_override_declaration(routes.get(runner_id, {})))
    return (
        f"{component_id} {option_id!r} cannot replace template "
        f"{template_id!r}'s {template_option!r} on runner {runner_id!r}: "
        + reason.rstrip(".") + ". Either "
        + ", or ".join(ways) + ".")


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _is_json_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def parameter_is_implemented(spec: Mapping[str, object]) -> bool:
    """Whether a declared parameter reaches a GPUWM runtime component.

    Implementation is proven by ``consuming_read`` -- a repo-relative file
    that actually reads the knob -- and never by an unbacked assertion.  A
    declaration carrying no citation makes no claim: it publishes the knob's
    type for discovery while leaving it unsettable, which is what lets each
    scheme's owner turn their own knobs on, by citing the read that makes it
    true, without editing anyone else's rows.

    ``implemented: false`` is the explicit roadmap form and additionally
    carries ``unimplemented_reason`` for a user interface to display.

    ``tools/check_parameter_claims.py`` is the gate: it re-resolves every
    citation, so a claim cannot outlive the code that justified it.
    """

    if spec.get("implemented") is False:
        return False
    citation = spec.get("consuming_read")
    return isinstance(citation, str) and bool(citation.strip())


def _parameter_error(spec: Mapping[str, object], value: object) -> str | None:
    if not parameter_is_implemented(spec):
        reason = spec.get("unimplemented_reason")
        detail = f": {reason}" if isinstance(reason, str) and reason else ""
        return f"is declared but not implemented{detail}"
    # A declared null default is the knob's "unset" value (diff_6th_factor2:
    # null takes the scheme's own factor), and validate_run_config accepts
    # it.  Refusing it made plan review refuse the registry's own default
    # on every route that allows the key (tests/test_authority_agreement.py,
    # 186 plans on tools.prepared_domain_tree_forecast).
    if value is None and "default" in spec and spec["default"] is None:
        return None
    expected = spec.get("type")
    if expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if valid:
            try:
                valid = math.isfinite(float(value))
            except (OverflowError, TypeError, ValueError):
                valid = False
    elif expected == "string":
        valid = isinstance(value, str)
    else:
        valid = _is_json_scalar(value)
    if not valid:
        return f"must have registry type {expected}"
    if "enum" in spec and value not in spec["enum"]:
        return f"must be one of {spec['enum']!r}"
    if "minimum" in spec:
        try:
            below_minimum = value < spec["minimum"]  # type: ignore[operator]
        except (TypeError, ValueError):
            return f"must be comparable with minimum {spec['minimum']}"
        if below_minimum:
            return f"must be >= {spec['minimum']}"
    if "maximum" in spec:
        try:
            above_maximum = value > spec["maximum"]  # type: ignore[operator]
        except (TypeError, ValueError):
            return f"must be comparable with maximum {spec['maximum']}"
        if above_maximum:
            return f"must be <= {spec['maximum']}"
    return None


def _same_value(left: object, right: object) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def _conditional_refusals(constraints: Mapping[str, object]) -> list[dict]:
    """``constraints.refused_when``, the CONDITIONAL refusal rules.

    The other three constraint kinds each ask ONE question:
    ``required_settings`` about a setting, ``forbidden_setting_values``
    about a setting's value, ``requires_components`` about a sibling
    component.  A real coupling is sometimes a conjunction across those
    kinds, and until 1.9 the registry could not say one: mp_physics=9 is
    refused with RTE+RRTMGP only when the RTE+RRTMGP ADAPTER is selected,
    and the adapter is chosen by the ``ra_rrtmg_variant`` PARAMETER over
    the same radiation component option that the legacy adapter uses.  The
    coupling was therefore written as prose in ``extensions`` -- which
    nothing evaluates -- and the registry told launchers a configuration
    was startable that ``validate_run_config`` refused, 320 combinations
    of it (tests/test_authority_agreement.py).

    A rule is a mapping with a human ``reason`` and any of:

    ``components``
        ``{component_id: [option_id, ...]}`` -- every named component must
        resolve to one of the listed options.
    ``settings``
        ``{setting: [value, ...]}`` -- every named setting must resolve to
        one of the listed values.  A setting the plan does not carry falls
        back to the registry's own declared default for that parameter, so
        a rule cannot be dodged by leaving the knob at its default.
    ``sources``
        ``[source_id, ...]`` -- the plan's ``context.source_id`` must be
        one of them (audit R-005).  A per-source incompatibility is a
        real thing -- native HRRR cannot retain its analyzed
        QC/QR/QI/QS/QG with microphysics off, and no radiation-only
        analyzed-cloud carrier is implemented -- and it was previously
        spelled only in prose inside a runner, where plan review never
        reached it and the run died in ``woof/ingest/real.py``.  It is a
        row now, so a source that gains such an incompatibility is table
        work.

    A rule MAY also carry the way out, in both halves:

    ``remedy_label``
        the sentence that names it, printed after ``reason`` wherever this
        refusal is shown, so the gate law's "and how to get out of it" is
        one string in one table rather than each door's own wording.
    ``remedy_settings``
        ``{setting: value}`` -- the smallest edit that clears the rule,
        machine-applicable.  A front end offering a repair applies this
        instead of parsing the prose for it, which is what
        woof/companion_physics.py used to do with a substring of one
        scheme's refusal.

    An empty rule fires on everything, which is never what anyone means,
    so it is treated as no rule at all rather than as an unconditional
    refusal.
    """

    rules = constraints.get("refused_when", [])
    if not isinstance(rules, list):
        return []
    return [
        rule for rule in rules
        if isinstance(rule, dict)
        and isinstance(rule.get("reason"), str)
        and (isinstance(rule.get("components"), dict)
             or isinstance(rule.get("settings"), dict)
             or isinstance(rule.get("sources"), list))
    ]


def conditional_refusal_sentence(rule: Mapping[str, object]) -> str:
    """One rule's refusal as a reader meets it: the breakage, then the way out.

    Split in the table (``reason`` and ``remedy_label``) and joined here,
    for the reason the gate law gives: a refusal without its way out is
    half a refusal, and a way out spelled once per door drifts from the
    refusal it belongs to.  A rule that declares no remedy prints its
    reason alone rather than an invented sentence.
    """

    reason = str(rule.get("reason", "")).strip()
    remedy = rule.get("remedy_label")
    if isinstance(remedy, str) and remedy.strip():
        return f"{reason} {remedy.strip()}".strip()
    return reason


def conditional_refusal_remedy(
    rule: Mapping[str, object],
) -> dict[str, object] | None:
    """One rule's machine-applicable way out, or ``None``.

    ``{setting: value}``, the smallest edit that clears the rule.  Kept
    beside :func:`conditional_refusal_sentence` so a caller that offers
    the remedy and a caller that prints it read one field each from one
    table.
    """

    remedy = rule.get("remedy_settings")
    if isinstance(remedy, Mapping) and remedy:
        return dict(remedy)
    return None


def _effective_setting(
    name: str,
    settings: Mapping[str, object],
    parameter_specs: Mapping[str, object],
) -> object:
    """A setting's resolved value, or the registry's declared default."""

    if name in settings:
        return settings[name]
    spec = parameter_specs.get(name)
    return spec.get("default") if isinstance(spec, Mapping) else None


def _conditional_refusal_fires(
    rule: Mapping[str, object],
    resolved_components: Mapping[str, str],
    settings: Mapping[str, object],
    parameter_specs: Mapping[str, object],
    source_id: object = None,
) -> bool:
    """Does every clause of one :func:`_conditional_refusals` rule hold?"""

    required_sources = rule.get("sources")
    if isinstance(required_sources, list):
        if source_id not in required_sources:
            return False
    required_components = rule.get("components")
    if isinstance(required_components, Mapping):
        for component_id, option_ids in required_components.items():
            if not isinstance(option_ids, list):
                return False
            if resolved_components.get(component_id) not in option_ids:
                return False
    required_settings = rule.get("settings")
    if isinstance(required_settings, Mapping):
        for name, values in required_settings.items():
            if not isinstance(values, list):
                return False
            observed = _effective_setting(name, settings, parameter_specs)
            if not any(_same_value(observed, value) for value in values):
                return False
    return True


def _empty_validation(
    registry: Mapping[str, object], plan: object
) -> dict[str, object]:
    try:
        plan_hash = canonical_sha256(plan)
    except (TypeError, ValueError):
        plan_hash = None
    return {
        "schema": VALIDATION_SCHEMA,
        "launchable": False,
        "errors": [],
        "warnings": [],
        # Machine questions, reported beside the plan's own verdict.
        # Every entry's code is in INSTALL_STATE_CODES.
        "install_state": [],
        "registry_sha256": registry_sha256(registry),
        "plan_sha256": plan_hash,
        "plan_id": None,
        "context": None,
        "acknowledgements": [],
        "acknowledgement_provenance": {},
        "resolved_domains": [],
        "asset_requirements": [],
    }


def validate_physics_plan(
    plan: object,
    *,
    registry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve a v2 plan without importing or executing any physics code.

    The one filesystem question it does ask is whether each selected
    option's declared table set is installed
    (:func:`resolve_asset_requirement`, which imports only
    :mod:`woof.data_assets` -- path resolution, no scheme). Collecting
    the requirements and never resolving them is what let an install
    missing a table pass plan review and refuse at load.

    THE ANSWER GOES IN ``install_state``, NOT IN ``errors``.  ``errors``
    and the ``launchable`` verdict they decide are about the PLAN, which
    is portable: the same document has to earn the same verdict on the
    laptop that authored it and on the node that runs it.  What is
    installed here is a machine question -- every code in
    :data:`INSTALL_STATE_CODES` -- and it is reported with what is
    missing, where it was looked for, and the command that stages it,
    while the door about to load the scheme is what refuses (see that
    constant's note for the door per code).  A plan that is launchable
    on a machine short of a table set therefore still reads
    ``launchable: true``, and its ``install_state`` is not empty.
    """

    selected_registry = physics_registry() if registry is None else deepcopy(registry)
    result = _empty_validation(selected_registry, plan)
    errors: list[dict[str, str]] = result["errors"]  # type: ignore[assignment]
    warnings: list[dict[str, str]] = result["warnings"]  # type: ignore[assignment]
    install_state: list[dict[str, str]] = result[  # type: ignore[assignment]
        "install_state"]

    if selected_registry.get("schema") != REGISTRY_SCHEMA:
        errors.append(
            _issue(
                "registry-schema",
                "registry.schema",
                f"expected {REGISTRY_SCHEMA!r}",
            )
        )
        return result
    if not isinstance(plan, dict):
        errors.append(_issue("plan-type", "$", "physics plan must be a JSON object"))
        return result
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append(
            _issue("plan-schema", "schema", f"expected {PLAN_SCHEMA!r}")
        )

    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id.strip():
        errors.append(
            _issue("plan-id", "plan_id", "must be a non-empty string")
        )
    else:
        result["plan_id"] = plan_id

    bound_registry_sha256 = plan.get("registry_sha256")
    actual_registry_sha256 = result["registry_sha256"]
    if not isinstance(bound_registry_sha256, str) or not bound_registry_sha256:
        errors.append(
            _issue(
                "registry-binding",
                "registry_sha256",
                "must bind the exact GPUWM physics registry semantic SHA-256",
            )
        )
    elif bound_registry_sha256 != actual_registry_sha256:
        errors.append(
            _issue(
                "stale-registry-binding",
                "registry_sha256",
                "plan registry SHA-256 differs from the active GPUWM registry",
            )
        )

    context = plan.get("context")
    context_valid = True
    if not isinstance(context, dict):
        errors.append(_issue("context-type", "context", "must be a JSON object"))
        context = {}
        context_valid = False
    normalized_context: dict[str, object] = {
        "source_id": None,
        "runner_id": None,
        "topology_id": None,
        "edges": [],
    }
    for key in ("source_id", "runner_id", "topology_id"):
        value = context.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                _issue("context-field", f"context.{key}", "must be a non-empty string")
            )
            context_valid = False
        else:
            normalized_context[key] = value
    if context_valid:
        result["context"] = normalized_context

    routes = selected_registry.get("runner_routes")
    routes = routes if isinstance(routes, dict) else {}
    runner_id = normalized_context.get("runner_id")
    route = routes.get(runner_id) if isinstance(runner_id, str) else None
    if runner_id is not None and not isinstance(route, dict):
        errors.append(
            _issue(
                "unknown-runner",
                "context.runner_id",
                f"runner {runner_id!r} is not registered",
            )
        )
    elif isinstance(route, dict):
        if route.get("implemented") is not True:
            errors.append(
                _issue(
                    "unimplemented-runner",
                    "context.runner_id",
                    f"runner {runner_id!r} is registered but not implemented",
                )
            )
        source_ids = route.get("source_ids", [])
        source_id = normalized_context.get("source_id")
        if (
            isinstance(source_id, str)
            and isinstance(source_ids, list)
            and "*" not in source_ids
            and source_id not in source_ids
        ):
            errors.append(
                _issue(
                    "unsupported-source-route",
                    "context.source_id",
                    f"runner {runner_id!r} does not register source {source_id!r}",
                )
            )
        topology_ids = route.get("topology_ids", [])
        topology_id = normalized_context.get("topology_id")
        if (
            isinstance(topology_id, str)
            and isinstance(topology_ids, list)
            and topology_id not in topology_ids
        ):
            errors.append(
                _issue(
                    "unsupported-topology-route",
                    "context.topology_id",
                    f"runner {runner_id!r} does not register topology {topology_id!r}",
                )
            )

    domains = plan.get("domains")
    if not isinstance(domains, list) or not domains:
        errors.append(
            _issue("domains-type", "domains", "must be a non-empty JSON array")
        )
        domains = []

    domain_ids: list[str] = []
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            continue
        domain_id = domain.get("domain_id")
        if isinstance(domain_id, str) and domain_id.strip():
            domain_ids.append(domain_id)
    duplicate_ids = sorted(
        domain_id for domain_id in set(domain_ids) if domain_ids.count(domain_id) > 1
    )
    for domain_id in duplicate_ids:
        errors.append(
            _issue("duplicate-domain", "domains", f"duplicate domain_id {domain_id!r}")
        )

    edges = plan.get("edges", [])
    if not isinstance(edges, list):
        errors.append(_issue("edges-type", "edges", "must be a JSON array"))
        edges = []
    normalized_edges: list[dict[str, str]] = []
    edge_pairs: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        path = f"edges[{index}]"
        if not isinstance(edge, dict):
            errors.append(_issue("edge-type", path, "must be a JSON object"))
            continue
        parent = edge.get("parent_domain_id")
        child = edge.get("child_domain_id")
        if (
            not isinstance(parent, str)
            or not parent.strip()
            or not isinstance(child, str)
            or not child.strip()
        ):
            errors.append(
                _issue(
                    "edge-field",
                    path,
                    "parent_domain_id and child_domain_id must be non-empty strings",
                )
            )
            continue
        if parent not in domain_ids or child not in domain_ids:
            errors.append(_issue("edge-domain", path, "edge references an unknown domain"))
            continue
        if parent == child:
            errors.append(_issue("edge-self", path, "domain cannot parent itself"))
            continue
        pair = (parent, child)
        if pair in edge_pairs:
            errors.append(
                _issue("duplicate-edge", path, f"duplicate edge {parent!r}->{child!r}")
            )
            continue
        edge_pairs.add(pair)
        normalized_edges.append(
            {"parent_domain_id": parent, "child_domain_id": child}
        )
    normalized_context["edges"] = normalized_edges

    topology_id = normalized_context.get("topology_id")
    if topology_id == "single-domain-v1" and len(domains) != 1:
        errors.append(
            _issue(
                "topology-domain-count",
                "domains",
                "single-domain-v1 requires exactly one domain",
            )
        )
    if topology_id == "single-domain-v1" and edges:
        errors.append(
            _issue(
                "single-domain-edges",
                "edges",
                "single-domain-v1 requires zero edges",
            )
        )
    if topology_id == "one-way-nested-v1" and len(domains) < 2:
        errors.append(
            _issue(
                "topology-domain-count",
                "domains",
                "one-way-nested-v1 requires at least two domains",
            )
        )
    if topology_id == "one-way-nested-v1" and domains:
        expected_edges = len(domains) - 1
        if len(normalized_edges) != expected_edges:
            errors.append(
                _issue(
                    "tree-edge-count",
                    "edges",
                    f"one-way tree requires {expected_edges} unique edges",
                )
            )
        incoming = {domain_id: 0 for domain_id in domain_ids}
        children: dict[str, list[str]] = {domain_id: [] for domain_id in domain_ids}
        for edge in normalized_edges:
            parent = edge["parent_domain_id"]
            child = edge["child_domain_id"]
            incoming[child] += 1
            children[parent].append(child)
        for domain_id, count in sorted(incoming.items()):
            if count > 1:
                errors.append(
                    _issue(
                        "multiple-parents",
                        "edges",
                        f"domain {domain_id!r} has {count} parents",
                    )
                )
        roots = sorted(domain_id for domain_id, count in incoming.items() if count == 0)
        if len(roots) != 1:
            errors.append(
                _issue(
                    "tree-root-count",
                    "edges",
                    f"one-way tree requires one root, found {len(roots)}",
                )
            )
        visiting: set[str] = set()
        visited: set[str] = set()
        cycle = False

        def visit(domain_id: str) -> None:
            nonlocal cycle
            if domain_id in visiting:
                cycle = True
                return
            if domain_id in visited:
                return
            visiting.add(domain_id)
            for child_id in children.get(domain_id, []):
                visit(child_id)
            visiting.remove(domain_id)
            visited.add(domain_id)

        for domain_id in domain_ids:
            visit(domain_id)
        if cycle:
            errors.append(_issue("tree-cycle", "edges", "one-way tree is cyclic"))
        if len(roots) == 1:
            reachable: set[str] = set()
            pending = [roots[0]]
            while pending:
                domain_id = pending.pop()
                if domain_id in reachable:
                    continue
                reachable.add(domain_id)
                pending.extend(children.get(domain_id, []))
            missing = sorted(set(domain_ids) - reachable)
            if missing:
                errors.append(
                    _issue(
                        "tree-disconnected",
                        "edges",
                        f"domains are disconnected from root {roots[0]!r}: {missing}",
                    )
                )

    # Depth below the tree root, used to index a template's per-domain
    # override columns.  A single-domain plan is depth 0.  Cyclic graphs are
    # already reported above; the visited guard keeps this terminating.
    parent_of = {
        edge["child_domain_id"]: edge["parent_domain_id"]
        for edge in normalized_edges
    }
    domain_depths: dict[str, int] = {}
    for domain_id in domain_ids:
        depth = 0
        cursor = domain_id
        walked = {cursor}
        while cursor in parent_of:
            cursor = parent_of[cursor]
            if cursor in walked:
                depth = 0
                break
            walked.add(cursor)
            depth += 1
        domain_depths[domain_id] = depth

    components = selected_registry.get("components")
    components = components if isinstance(components, dict) else {}
    templates = selected_registry.get("templates")
    templates = templates if isinstance(templates, dict) else {}
    parameter_specs = selected_registry.get("parameters")
    parameter_specs = parameter_specs if isinstance(parameter_specs, dict) else {}
    warning_policy = selected_registry.get("warning_policy", {})
    warning_policy = warning_policy if isinstance(warning_policy, dict) else {}
    # Both tiers come off the ladder, not out of warning_policy: the policy
    # block is the document's readable echo of them and the loader refuses a
    # registry where the two disagree, so reading the ladder here leaves
    # exactly one ordering of maturity names in this module.
    warn_maturities = _ladder_tier(selected_registry, "warn")
    nonwarning_maturities = _ladder_tier(selected_registry, "nonwarning")
    warn_unknown_maturity = (
        warning_policy.get("unknown_implemented_maturity") == "warn"
    )

    selector_owners: dict[str, str] = {}
    for component_id, component in components.items():
        if not isinstance(component, dict):
            continue
        keys = component.get("selector_keys", [])
        if not isinstance(keys, list):
            continue
        for key in keys:
            if isinstance(key, str):
                selector_owners[key] = component_id

    route_mode = route.get("mode") if isinstance(route, dict) else None
    route_component_overrides = set(
        route.get("allowed_component_overrides", [])
        if isinstance(route, dict)
        else []
    )
    route_component_options: dict[str, set[str]] = {}
    raw_component_options = (
        route.get("allowed_component_options", {})
        if isinstance(route, dict)
        else {}
    )
    if isinstance(raw_component_options, dict):
        route_component_options = {
            component_id: {
                option_id for option_id in option_ids
                if isinstance(option_id, str)
            }
            for component_id, option_ids in raw_component_options.items()
            if isinstance(component_id, str) and isinstance(option_ids, list)
        }
    route_parameter_keys = set(
        route.get("allowed_parameter_keys", []) if isinstance(route, dict) else []
    )
    route_expert_settings = set(
        route.get("allowed_expert_setting_keys", [])
        if isinstance(route, dict)
        else []
    )
    route_expert_selectors = set(
        route.get("allowed_expert_selector_keys", [])
        if isinstance(route, dict)
        else []
    )
    # A route's template lists are the reachability declaration the GUI reads.
    # ``route_declares_templates`` records that the route publishes any list at
    # all: a route that does must be exhaustive, so a source it registers with
    # NO list reaches no template rather than every template.  It used to reach
    # every template -- ``source_template_ids`` named only ``hrrr`` on the tree
    # route, so an era5/gfs/20crv3 tree plan could name any registered
    # template, which made "is this option reachable" unanswerable from the
    # declaration and made an expert-only template reachable without its
    # acknowledgement through a source that simply had no list.
    route_source_templates: set[str] = set()
    route_expert_templates: set[str] = set()
    # Template id -> the concrete breakage that keeps it off THIS route,
    # with the way out, as tools/build_registry.py wrote it.  A route that
    # publishes one of these is not merely silent about the template: it
    # has stated that its runner refuses the template at the door, so plan
    # review refuses it here instead, which is where a refusal belongs.
    route_refused_templates: Mapping[str, str] = {}
    route_declares_templates = False
    route_expert_acknowledgement = None
    if isinstance(route, dict):
        refused_templates = route.get("refused_template_ids", {})
        if isinstance(refused_templates, Mapping):
            route_refused_templates = {
                template_id: reason
                for template_id, reason in refused_templates.items()
                if isinstance(template_id, str) and isinstance(reason, str)
            }
        source_templates = route.get("source_template_ids", {})
        expert_templates = route.get("expert_template_ids", {})
        source_id = normalized_context.get("source_id")
        if isinstance(source_templates, dict):
            route_declares_templates = bool(source_templates)
            if isinstance(source_id, str):
                declared = source_templates.get(source_id, [])
                if isinstance(declared, list):
                    route_source_templates = {
                        value for value in declared if isinstance(value, str)
                    }
        if isinstance(expert_templates, dict):
            route_declares_templates = (
                route_declares_templates or bool(expert_templates))
            if isinstance(source_id, str):
                route_expert_templates = set(
                    expert_template_ids_for_source(route, source_id))
        acknowledgement = route.get("expert_acknowledgement_id")
        if isinstance(acknowledgement, str) and acknowledgement.strip():
            route_expert_acknowledgement = acknowledgement

    # Consent is plan-level and names the route's own acknowledgement id, so a
    # launcher cannot satisfy it by accident: an unrecognized id is an error
    # rather than a no-op, and consent without an expert template is reported
    # too (it means the plan and the route disagree about what is being run).
    acknowledged: set[str] = set()
    plan_acknowledgements = plan.get("acknowledgements", [])
    if not isinstance(plan_acknowledgements, list):
        errors.append(
            _issue(
                "expert-acknowledgements-type",
                "acknowledgements",
                "must be a JSON array of route acknowledgement ids",
            )
        )
        plan_acknowledgements = []
    for index, value in enumerate(plan_acknowledgements):
        if not isinstance(value, str) or not value.strip():
            errors.append(
                _issue(
                    "expert-acknowledgement-id",
                    f"acknowledgements[{index}]",
                    "must be a non-empty string",
                )
            )
            continue
        if value != route_expert_acknowledgement:
            errors.append(
                _issue(
                    "unknown-expert-acknowledgement",
                    f"acknowledgements[{index}]",
                    f"runner {runner_id!r} does not publish acknowledgement "
                    f"{value!r}",
                )
            )
            continue
        acknowledged.add(value)
    result["acknowledgements"] = sorted(acknowledged)
    result["acknowledgement_provenance"] = {
        value: ["physics-plan.acknowledgements"]
        for value in sorted(acknowledged)
    }

    # Keep issue anchors in domain-index space and aggregate requirements
    # in public grid-id space.
    asset_domains: dict[
        str, tuple[dict[str, object], list[str], list[int]]] = {}
    resolved_domains: list[dict[str, object]] = result[  # type: ignore[assignment]
        "resolved_domains"
    ]
    for index, domain in enumerate(domains):
        base_path = f"domains[{index}]"
        if not isinstance(domain, dict):
            errors.append(_issue("domain-type", base_path, "must be a JSON object"))
            continue
        domain_id = domain.get("domain_id")
        if not isinstance(domain_id, str) or not domain_id.strip():
            errors.append(
                _issue("domain-id", f"{base_path}.domain_id", "must be a non-empty string")
            )
            domain_id = f"invalid-domain-{index}"

        template_id = canonical_template_id(domain.get("template_id"), selected_registry)
        template: Mapping[str, Any] = {}
        if template_id is not None:
            if not isinstance(template_id, str):
                errors.append(
                    _issue(
                        "template-id",
                        f"{base_path}.template_id",
                        "must be a string",
                    )
                )
            elif not isinstance(templates.get(template_id), dict):
                errors.append(
                    _issue(
                        "unknown-template",
                        f"{base_path}.template_id",
                        f"template {template_id!r} is not registered",
                    )
                )
            else:
                template = templates[template_id]
                refusal = route_refused_templates.get(template_id)
                if refusal is not None:
                    # NOT the evidence warning below.  That one covers a
                    # template the route says nothing about, which still
                    # runs; this one covers a template the route has
                    # declared its runner cannot build a product for, and
                    # it fires at review rather than as a bare
                    # ``unsupported physics profile`` after the operator
                    # has paid for preparation.
                    errors.append(
                        _issue(
                            "template-refused-on-route",
                            f"{base_path}.template_id",
                            f"runner {runner_id!r} does not offer template "
                            f"{template_id!r}: {refusal}",
                        )
                    )
                elif route_declares_templates and template_id not in (
                        route_source_templates | route_expert_templates):
                    warnings.append(
                        _issue(
                            "template-route-evidence",
                            f"{base_path}.template_id",
                            f"template {template_id!r} has no declared "
                            "evidence entry for this source/runner route; "
                            "the resolved runtime settings still apply",
                        )
                    )
                elif template_id in route_expert_templates:
                    if route_expert_acknowledgement is None:
                        errors.append(
                            _issue(
                                "expert-route-policy",
                                "registry.runner_routes",
                                f"runner {runner_id!r} offers expert templates "
                                "without publishing expert_acknowledgement_id",
                            )
                        )
                    elif route_expert_acknowledgement not in acknowledged:
                        warnings.append(
                            _issue(
                                "expert-acknowledgement-advisory",
                                f"{base_path}.template_id",
                                f"template {template_id!r} carries an "
                                f"unacknowledged evidence advisory; it can "
                                f"run. To acknowledge it, add "
                                f"acknowledgements "
                                f"[{route_expert_acknowledgement!r}]",
                            )
                        )
                    for message in route.get("expert_warnings", []) or []:
                        if isinstance(message, str):
                            warnings.append(
                                _issue(
                                    "expert-template-warning",
                                    f"{base_path}.template_id",
                                    message,
                                )
                            )
                template_maturity = canonical_maturity(
                    template.get("maturity"), selected_registry)
                if template_maturity in warn_maturities or (
                    warn_unknown_maturity
                    and template_maturity not in nonwarning_maturities
                ):
                    warnings.append(
                        _issue(
                            "template-maturity",
                            f"{base_path}.template_id",
                            f"template {template_id!r} maturity is {template_maturity}; warning does not block launch",
                        )
                    )
                template_warnings = template.get("warnings", [])
                if isinstance(template_warnings, list):
                    for message in template_warnings:
                        if isinstance(message, str):
                            warnings.append(
                                _issue(
                                    "template-warning",
                                    f"{base_path}.template_id",
                                    message,
                                )
                            )

        resolved_components: dict[str, str] = {}
        template_components = template.get("components", {})
        if isinstance(template_components, dict):
            resolved_components.update(template_components)
        requested_components = domain.get("components", {})
        if not isinstance(requested_components, dict):
            errors.append(
                _issue(
                    "components-type",
                    f"{base_path}.components",
                    "must be a JSON object",
                )
            )
            requested_components = {}
        if (
            isinstance(route, dict)
            and route.get("require_explicit_components") is True
            and not requested_components
        ):
            errors.append(
                _issue(
                    "explicit-components-required",
                    f"{base_path}.components",
                    "runner route requires a non-empty explicit component map",
                )
            )
        for component_id, option_id in requested_components.items():
            if component_id not in components:
                errors.append(
                    _issue(
                        "unknown-component",
                        f"{base_path}.components.{component_id}",
                        f"component {component_id!r} is not registered",
                    )
                )
                continue
            if not isinstance(option_id, str):
                errors.append(
                    _issue(
                        "option-id",
                        f"{base_path}.components.{component_id}",
                        "option id must be a string",
                    )
                )
                continue
            resolved_components[component_id] = option_id

        expert = domain.get("expert_overrides", {})
        if not isinstance(expert, dict):
            errors.append(
                _issue(
                    "expert-overrides-type",
                    f"{base_path}.expert_overrides",
                    "must be a JSON object",
                )
            )
            expert = {}
        unknown_expert_sections = sorted(set(expert) - {"selectors", "settings"})
        for section in unknown_expert_sections:
            errors.append(
                _issue(
                    "expert-overrides-section",
                    f"{base_path}.expert_overrides.{section}",
                    "only selectors and settings are recognized",
                )
            )
        expert_selectors = expert.get("selectors", {})
        expert_settings = expert.get("settings", {})
        if not isinstance(expert_selectors, dict):
            errors.append(
                _issue(
                    "expert-selectors-type",
                    f"{base_path}.expert_overrides.selectors",
                    "must be a JSON object",
                )
            )
            expert_selectors = {}
        if not isinstance(expert_settings, dict):
            errors.append(
                _issue(
                    "expert-settings-type",
                    f"{base_path}.expert_overrides.settings",
                    "must be a JSON object",
                )
            )
            expert_settings = {}
        for selector in sorted(expert_selectors):
            if selector not in selector_owners:
                errors.append(
                    _issue(
                        "unknown-expert-selector",
                        f"{base_path}.expert_overrides.selectors.{selector}",
                        "selector is not registered and cannot imply implementation",
                    )
                )
        for selector in sorted(set(expert_settings) & set(selector_owners)):
            errors.append(
                _issue(
                    "selector-in-settings",
                    f"{base_path}.expert_overrides.settings.{selector}",
                    "registered selectors must be placed in expert_overrides.selectors",
                )
            )

        # AUDIT R-059.  Both refusals named no breakage, and the message
        # that sends a user here ("use expert_overrides.settings for
        # passthrough") pointed at a door every route nailed shut.  The
        # route now states what its empty list protects, and the sentence
        # is printed verbatim so the refusal carries a reason and a way
        # out rather than a restatement of itself.
        expert_refusal = (
            route.get("expert_override_refusal_reason")
            if isinstance(route, dict) else None
        )
        expert_refusal = (
            f" {expert_refusal}"
            if isinstance(expert_refusal, str) and expert_refusal else "")
        for name in sorted(expert_selectors):
            if name not in route_expert_selectors:
                errors.append(
                    _issue(
                        "expert-selector-route",
                        f"{base_path}.expert_overrides.selectors.{name}",
                        "runner route does not accept this expert selector."
                        + expert_refusal,
                    )
                )
        for name in sorted(expert_settings):
            if name not in route_expert_settings:
                errors.append(
                    _issue(
                        "expert-setting-route",
                        f"{base_path}.expert_overrides.settings.{name}",
                        "runner route does not accept this expert setting."
                        + expert_refusal,
                    )
                )

        # A route refusal names the option it turns away, what the route
        # declares instead, and the way to the option: the templates on
        # this route and source that carry it, or the runner that admits
        # per-domain overrides.  "runner route does not allow this
        # component option to vary per domain" named none of the three
        # and left a reader to guess whether the scheme was missing, the
        # pairing illegal or the route closed by declaration.
        #
        # AUDIT R-021.  ONE question is asked for every route mode, of
        # the route's own declaration, where a separate fixed-template
        # branch used to refuse EVERY component key outright.  That
        # blanket branch described no runner -- this route's runner had
        # its profile whitelist removed by the 2026-07-31 ruling and runs
        # any engine-valid suite -- and it refused a component map that
        # merely RESTATED the base template's own composition, which
        # overrides nothing and so can break nothing.  A route that
        # declares no overrides still refuses them all; the two modes
        # differ only in which sentence says what the closure protects.
        reachable_templates = (
            (route_source_templates | route_expert_templates)
            or set(templates))
        if isinstance(template_components, dict):
            for component_id, option_id in sorted(requested_components.items()):
                if not isinstance(option_id, str):
                    continue
                component = components.get(component_id)
                options = (component.get("options", {})
                           if isinstance(component, dict) else {})
                option = (options.get(option_id)
                          if isinstance(options, dict) else None)
                if (not isinstance(option, dict)
                        or option.get("implemented") is not True):
                    # Recognition and implementation have their own errors.
                    # An unknown option cannot imply a route restriction.
                    continue
                if (
                    template_components.get(component_id) == option_id
                    or component_id in route_component_overrides
                    or option_id in route_component_options.get(
                        component_id, set())
                ):
                    continue
                errors.append(
                    _issue(
                        "component-override-route",
                        f"{base_path}.components.{component_id}",
                        _fixed_template_override_refusal(
                            runner_id, routes, templates, reachable_templates,
                            template_id, template_components,
                            component_id, option_id,
                            route_expert_templates,
                            route_expert_acknowledgement)
                        if route_mode == "fixed-template"
                        else _override_route_refusal(
                            runner_id, routes,
                            normalized_context.get("source_id"),
                            templates, reachable_templates, template_id,
                            template_components, component_id, option_id,
                            route_component_overrides,
                            route_component_options,
                            route_expert_templates,
                            route_expert_acknowledgement),
                    )
                )

        for component_id in sorted(components):
            component = components[component_id]
            if not isinstance(component, dict):
                continue
            option_id = resolved_components.get(component_id)
            if not isinstance(option_id, str):
                errors.append(
                    _issue(
                        "missing-component",
                        f"{base_path}.components.{component_id}",
                        "select an option directly or through a template",
                    )
                )
                continue
            options = component.get("options", {})
            if not isinstance(options, dict) or not isinstance(options.get(option_id), dict):
                errors.append(
                    _issue(
                        "unknown-option",
                        f"{base_path}.components.{component_id}",
                        f"option {option_id!r} is not registered for {component_id}. "
                        f"Check its spelling against components.{component_id}.options "
                        "in the physics registry; changing runner cannot register it.",
                    )
                )
                continue
            selected_option = options[option_id]
            keys = component.get("selector_keys", [])
            if isinstance(keys, list) and selected_option.get("implemented") is True:
                projection = dict(selected_option.get("selectors", {}))
                touched = False
                for key in keys:
                    if key in expert_selectors:
                        projection[key] = expert_selectors[key]
                        touched = True
                if touched:
                    matches = [
                        (candidate_id, candidate)
                        for candidate_id, candidate in options.items()
                        if isinstance(candidate, dict)
                        and all(
                            key in candidate.get("selectors", {})
                            and _same_value(candidate["selectors"][key], projection.get(key))
                            for key in keys
                        )
                    ]
                    if len(matches) != 1:
                        errors.append(
                            _issue(
                                "unknown-selector-combination",
                                f"{base_path}.expert_overrides.selectors",
                                f"selectors for {component_id} do not identify one registered option",
                            )
                        )
                    else:
                        candidate_id, candidate = matches[0]
                        if candidate.get("implemented") is not True:
                            errors.append(
                                _issue(
                                    "unimplemented-selector",
                                    f"{base_path}.expert_overrides.selectors",
                                    f"selectors resolve to unimplemented {component_id} option {candidate_id!r}",
                                )
                            )
                        else:
                            if candidate_id != option_id:
                                warnings.append(
                                    _issue(
                                        "expert-selector-resolution",
                                        f"{base_path}.expert_overrides.selectors",
                                        f"expert selectors replace {component_id} option {option_id!r} with {candidate_id!r}",
                                    )
                                )
                            resolved_components[component_id] = candidate_id

        settings: dict[str, object] = {
            name: spec["default"]
            for name, spec in parameter_specs.items()
            if isinstance(spec, dict)
            and "default" in spec
            and parameter_is_implemented(spec)
        }
        template_parameters = template.get("parameters", {})
        # The template's own values, kept so they can be re-applied over
        # the component options below: a template that pins a knob its
        # component also sets (the GSD MYNN 4.1 suite's bl_mynn_mixlength
        # 2 and scalar_pblmix 1 against the mynn option's 1 and 0) must
        # win, or the suite silently runs the component's value.
        template_layer: dict[str, object] = {}
        if isinstance(template_parameters, dict):
            settings.update(template_parameters)
            template_layer.update(template_parameters)
        # Per-domain template values, indexed by depth below the tree root.
        # Several WRF knobs are max_domains in the Registry and were varied
        # down the nest chain by the verified runs (sixth-order diffusion
        # factor, radiation interval, acoustic off-centering).  A template
        # records those columns verbatim rather than deriving them from grid
        # spacing, because they came from a namelist, not from a scaling law.
        overrides = template.get("per_domain_overrides")
        if isinstance(overrides, list):
            depth = domain_depths.get(domain_id)
            if depth is not None and depth < len(overrides):
                column = overrides[depth]
                if isinstance(column, dict):
                    for name, value in column.items():
                        if name == "nominal_dx_m":
                            continue
                        spec = parameter_specs.get(name)
                        message = (
                            _parameter_error(spec, value)
                            if isinstance(spec, dict)
                            else "parameter is not registered"
                        )
                        if message is not None:
                            errors.append(
                                _issue(
                                    "template-per-domain-override",
                                    f"registry.templates.{template_id}"
                                    f".per_domain_overrides[{depth}].{name}",
                                    message,
                                )
                            )
                            continue
                        settings[name] = value
                        template_layer[name] = value
        domain_parameters = domain.get("parameters", {})
        if not isinstance(domain_parameters, dict):
            errors.append(
                _issue(
                    "parameters-type",
                    f"{base_path}.parameters",
                    "must be a JSON object",
                )
            )
            domain_parameters = {}
        # A route that leaves a loader-accepted knob out of
        # allowed_parameter_keys publishes WHY in deferred_parameter_keys,
        # naming the component the knob belongs to and the way to the
        # value.  The reason was written by tools/build_registry.py and
        # read by nothing, so the refusal a user actually met was
        # "runner route does not accept this per-domain setting" and
        # stopped there, while the release surface promised the reason.
        deferred_parameters = (
            route.get("deferred_parameter_keys") if isinstance(route, dict)
            else None)
        if not isinstance(deferred_parameters, dict):
            deferred_parameters = {}
        for name in sorted(domain_parameters):
            if name not in route_parameter_keys:
                deferred = deferred_parameters.get(name)
                errors.append(
                    _issue(
                        "parameter-route",
                        f"{base_path}.parameters.{name}",
                        "runner route does not accept this per-domain setting"
                        + _deferred_parameter_clause(
                            runner_id, deferred, route_parameter_keys),
                    )
                )

        # Parameters an option of a CHANGED component set: the override's
        # own values, which the template layer re-applied below must not
        # take back.  b9a01da37 re-applies the template over its UNCHANGED
        # components' option defaults (the GSD MYNN 4.1 suite's pins); over
        # a changed component it handed the replaced scheme's receipt token
        # back to the replacement -- a radiation override away from the
        # RRTMG 4/4 pair kept wrf_rrtmg_compatibility, passed plan review,
        # and was refused at run start.
        overridden: set[str] = set()
        for component_id in sorted(components):
            option_id = resolved_components.get(component_id)
            component = components.get(component_id)
            options = component.get("options", {}) if isinstance(component, dict) else {}
            option = options.get(option_id) if isinstance(options, dict) else None
            if not isinstance(option, dict):
                continue
            if option.get("implemented") is not True:
                errors.append(
                    _issue(
                        "unimplemented-option",
                        f"{base_path}.components.{component_id}",
                        f"option {option_id!r} is registered but not implemented",
                    )
                )
                continue
            maturity = canonical_maturity(option.get("maturity"), selected_registry)
            if maturity in warn_maturities or (
                warn_unknown_maturity and maturity not in nonwarning_maturities
            ):
                warnings.append(
                    _issue(
                        "maturity",
                        f"{base_path}.components.{component_id}",
                        f"{option_id} maturity is {maturity}; warning does not block launch",
                    )
                )
            option_warnings = option.get("warnings", [])
            if isinstance(option_warnings, list):
                for message in option_warnings:
                    if isinstance(message, str):
                        warnings.append(
                            _issue(
                                "component-warning",
                                f"{base_path}.components.{component_id}",
                                message,
                            )
                        )
            selectors = option.get("selectors", {})
            parameters = option.get("parameters", {})
            if isinstance(selectors, dict):
                settings.update(selectors)
            if isinstance(parameters, dict):
                # An unchanged component keeps the named template's explicit
                # settings. Generic option defaults must not silently select
                # another trajectory for a composed template.
                for name, value in parameters.items():
                    if (template_components.get(component_id) != option_id
                            or name not in template_parameters):
                        settings[name] = value
                    if template_components.get(component_id) != option_id:
                        overridden.add(name)
            requirements = option.get("asset_requirements", [])
            if isinstance(requirements, list):
                for requirement in requirements:
                    if not isinstance(requirement, dict):
                        continue
                    key = canonical_json(requirement)
                    if key not in asset_domains:
                        asset_domains[key] = (deepcopy(requirement), [], [])
                    if domain_id not in asset_domains[key][1]:
                        asset_domains[key][1].append(domain_id)
                    if index not in asset_domains[key][2]:
                        asset_domains[key][2].append(index)

        # Template, then per-depth template columns, over component options
        # the template itself carries; explicit domain parameters below
        # still win over both.
        settings.update({name: value for name, value in template_layer.items()
                         if name not in overridden})
        # A template knob the overrides leave unread returns to its default
        # (U-E4): the GSD MYNN 4.1 suite's thompson_version='wrf_39_noaa'
        # under any microphysics but mp_physics=28, or its gsd_41 MYNN
        # generation under another PBL, named a scheme the plan no longer
        # runs and the run door refused it.  Only when a component is
        # overridden, so a template as written resolves exactly as before.
        if any(template_components.get(component_id) != option_id
               for component_id, option_id in resolved_components.items()):
            for _ in range(len(template_layer)):
                reverted = False
                for name in template_layer:
                    spec = parameter_specs.get(name)
                    if (name in overridden or not isinstance(spec, dict)
                            or "default" not in spec
                            or settings.get(name) == spec["default"]
                            or registry_knob_is_read(name, settings, selected_registry)):
                        continue
                    settings[name] = spec["default"]
                    reverted = True
                if not reverted:
                    break

        for name, value in domain_parameters.items():
            spec = parameter_specs.get(name)
            if not isinstance(spec, dict):
                errors.append(
                    _issue(
                        "unknown-parameter",
                        f"{base_path}.parameters.{name}",
                        "parameter is not registered; use expert_overrides.settings for passthrough",
                    )
                )
                continue
            message = _parameter_error(spec, value)
            if message is not None:
                errors.append(
                    _issue("parameter-value", f"{base_path}.parameters.{name}", message)
                )
                continue
            settings[name] = value

        for name, value in expert_settings.items():
            if name in selector_owners:
                continue
            if not _is_json_scalar(value):
                errors.append(
                    _issue(
                        "expert-setting-value",
                        f"{base_path}.expert_overrides.settings.{name}",
                        "expert setting must be a finite JSON scalar",
                    )
                )
                continue
            spec = parameter_specs.get(name)
            if isinstance(spec, dict):
                message = _parameter_error(spec, value)
                if message is not None:
                    errors.append(
                        _issue(
                            "expert-setting-value",
                            f"{base_path}.expert_overrides.settings.{name}",
                            message,
                        )
                    )
                    continue
            else:
                warnings.append(
                    _issue(
                        "untyped-expert-setting",
                        f"{base_path}.expert_overrides.settings.{name}",
                        "passed through without claiming registry implementation or validation",
                    )
                )
            settings[name] = value
        for name, value in expert_selectors.items():
            if name in selector_owners and _is_json_scalar(value):
                settings[name] = value

        # Some scheme-only controls must stay at their declared default
        # when their consumer is inactive, or the run would discard them.
        for name, spec in parameter_specs.items():
            if not isinstance(spec, dict) or spec.get("inactive_value_policy") != "require-default":
                continue
            if registry_knob_is_read(name, settings, selected_registry):
                continue
            value = _effective_setting(name, settings, parameter_specs)
            if value != spec.get("default"):
                errors.append(_issue(
                    "parameter-value", f"{base_path}.parameters.{name}",
                    f"{name}={value!r} needs {spec.get('read_when')}; "
                    "the inactive consumer would discard its nondefault value"))
        # A new parameter can declare the consumers that read a moved value.
        # Defaults remain inert outside them, preserving existing identities.
        for name, spec in parameter_specs.items():
            if not isinstance(spec, dict):
                continue
            required = spec.get("nondefault_requires_components")
            if not isinstance(required, dict):
                continue
            value = _effective_setting(name, settings, parameter_specs)
            if _same_value(value, spec.get("default")):
                continue
            for component, allowed in required.items():
                if resolved_components.get(component) not in allowed:
                    errors.append(_issue(
                        "parameter-consumer", f"{base_path}.parameters.{name}",
                        f"{name}={value!r} would be ignored: it requires "
                        f"{component} in {allowed!r}; leave it at its default "
                        "or select a consuming option"))

        # Use the run door's coupling law after parameter values resolve.
        from types import SimpleNamespace
        from woof.config import validate_fog_kfeta_config
        try:
            validate_fog_kfeta_config(SimpleNamespace(
                grav_settling=settings.get("grav_settling", 0),
                kfeta_trigger=settings.get("kfeta_trigger", 1),
                mp_physics=settings.get("mp_physics", 0),
                cu_physics=settings.get("cu_physics", 0)))
        except ValueError as exc:
            errors.append(_issue("fog-kfeta-pairing", base_path, str(exc)))
        from woof.core.madwrf_contract import MAD_DEFAULTS, validate_madwrf
        try:
            validate_madwrf(SimpleNamespace(
                **{name: settings.get(name, default)
                   for name, default in MAD_DEFAULTS.items()},
                mp_physics=settings.get("mp_physics", 0),
                use_mp_re=_effective_setting("use_mp_re", settings, parameter_specs)))
        except ValueError as exc:
            errors.append(_issue("madwrf-dependency", base_path, str(exc)))
        from woof.config import validate_noah_mosaic_config
        try:
            validate_noah_mosaic_config(SimpleNamespace(
                sf_surface_mosaic=settings.get("sf_surface_mosaic", 0),
                mosaic_cat=settings.get("mosaic_cat", 3),
                sf_surface_physics=settings.get("sf_surface_physics", 0),
                sf_urban_physics=settings.get("sf_urban_physics", 0),
                mosaic_urban_canopy=settings.get("mosaic_urban_canopy",
                                                 "dominant")))
        except ValueError as exc:
            errors.append(_issue("noah-mosaic-pairing", base_path, str(exc)))
        from woof.config import validate_irrigation_config
        irrigation_defaults = dict(sf_surf_irr_scheme=0, irr_daily_amount=0.0,
            irr_start_hour=0, irr_num_hours=0, irr_start_julianday=0,
            irr_end_julianday=0, irr_freq=1, irr_ph=0,
            sf_surface_physics=0, sf_surface_mosaic=0, sf_urban_physics=0,
            moist=False, mp_physics=0)
        try:
            validate_irrigation_config(SimpleNamespace(**{
                key: settings.get(key, default)
                for key, default in irrigation_defaults.items()}))
        except ValueError as exc:
            errors.append(_issue("irrigation-pairing", base_path, str(exc)))
        # The stochastic-pattern consumers, by the same law: spp_conv=1 and
        # spp_pbl=1 are offered on routes whose templates select no Grell-
        # Freitas or MYNN, and the run door refuses a pattern no scheme reads
        # (woof.config.validate_spp_config).  Plan review called 340 such
        # plans launchable (tests/test_authority_agreement.py).
        from woof.config import validate_spp_config
        try:
            validate_spp_config(SimpleNamespace(
                spp_conv=settings.get("spp_conv", 0),
                spp_pbl=settings.get("spp_pbl", 0),
                spp_lsm=settings.get("spp_lsm", 0),
                cu_physics=settings.get("cu_physics", 0),
                bl_pbl_physics=settings.get("bl_pbl_physics", 0),
                sf_sfclay_physics=settings.get("sf_sfclay_physics", 0),
                sf_surface_physics=settings.get("sf_surface_physics", 0)))
        except ValueError as exc:
            errors.append(_issue("spp-consumer", base_path, str(exc)))
        # Two more run-door couplings on the same footing.  A template that
        # writes scalar_pblmix=1 (the gsd41 HRRR template) offered every
        # microphysics override as launchable although the scalar mixing
        # reads Thompson-aerosol fields only, and offered spp_pbl=1 on its
        # gsd_41 MYNN, which has no stochastic kernels; both were refused
        # at the run door (36 plans, tests/test_authority_agreement.py).
        from woof.config import (validate_mynn_generation_spp,
                                  validate_scalar_pblmix_consumer)
        coupled = SimpleNamespace(
            scalar_pblmix=settings.get("scalar_pblmix", 0),
            bl_pbl_physics=settings.get("bl_pbl_physics", 0),
            mp_physics=settings.get("mp_physics", 0),
            bl_mynn_version=settings.get("bl_mynn_version", "wrf_461"),
            spp_pbl=settings.get("spp_pbl", 0))
        for law in (validate_scalar_pblmix_consumer,
                    validate_mynn_generation_spp):
            try:
                law(coupled)
            except (ValueError, NotImplementedError) as exc:
                errors.append(_issue("pbl-coupling", base_path, str(exc)))
        # The lightning package's producers, by the same law: the nested
        # route offers lightning_option as a free per-domain knob, and the
        # run door refuses options 1 and 2 without do_radar_ref=1, 11 without
        # a cumulus cloud top and 3 without a graupel scheme
        # (woof.lightning_contract.validate_lightning).  Plan review called
        # 602 such plans launchable on the 2.9.0 round-2 tree
        # (tests/test_authority_agreement.py), so it asks the run door's own
        # function rather than a copy of its producer tables.
        from woof.lightning_contract import (LIGHTNING_DEFAULTS,
                                              validate_lightning)
        try:
            validate_lightning(SimpleNamespace(
                **{name: settings.get(name, default)
                   for name, default in LIGHTNING_DEFAULTS.items()},
                moist=settings.get("moist", True),
                mp_physics=settings.get("mp_physics", 0),
                cu_physics=settings.get("cu_physics", 0)))
        except ValueError as exc:
            errors.append(_issue("lightning-dependency", base_path, str(exc)))

        # AUDIT R-005, and it runs HERE -- after every source of a setting
        # and before the constraint battery below -- because the value it
        # judges is part of the configuration the constraints are judged
        # against.
        #
        # This pass first tried to RESOLVE the value: plan review set
        # moist=true and warned that it had.  That was a resolution
        # nothing performs.  This function reports -- its only product
        # caller is woof/source_cli.py --validate-physics-plan, and
        # ``resolved_domains`` has no consumer -- while the RunConfig a
        # runner builds takes moist from the microphysics-off option's own
        # row (``moist: false``, the idealized reading) and from the
        # experiment config.  Review would have called such a plan
        # launchable and woof/ingest/real.py:``real initialization
        # requires cfg.moist=True`` would have raised before step 0, which
        # is the same drift the audit exists to close with the refusal
        # moved LATER rather than earlier.  So it stays an error, at
        # review, where it fires before any work is done.
        #
        # What the resolution attempt got right is that a remedy must be
        # expressible on the route it is prescribed to.  The old text
        # prescribed one that two of the three routes could not express;
        # each route now DECLARES where its moist value lives
        # (runner_routes.<runner>.moist_declaration_site) and the refusal
        # reads that declaration out rather than assuming a door.
        if (
            isinstance(route, dict)
            and route.get("requires_moist_real_initialization") is True
            and resolved_components.get("microphysics") == "off"
        ):
            site = route.get("moist_declaration_site")
            if not isinstance(site, str) or not site:
                site = "where this route accepts a physics value"
            # EXPLICIT means the plan said it: a per-domain parameter or an
            # expert setting.  The microphysics-off option's own row carries
            # moist=false, so an unstated plan and a stated dry column reach
            # the loader as the same RunConfig -- but they are different
            # mistakes and get different sentences.
            moist_setting = domain_parameters.get(
                "moist", expert_settings.get("moist"))
            if moist_setting is False:
                errors.append(
                    _issue(
                        "real-source-mp-off-requires-moist",
                        f"{base_path}.parameters.moist",
                        "moist=false is refused with microphysics off on "
                        "this real-source route: every source it registers "
                        "enters woof.ingest.real, which refuses a dry "
                        "column outright (`real initialization requires "
                        "cfg.moist=True`), and a dry column would drop the "
                        "analyzed water vapour the source supplies. Set "
                        f"moist=true {site}, or run a dry column on an "
                        "idealized composition.",
                    )
                )
            elif moist_setting is not True:
                errors.append(
                    _issue(
                        "real-source-mp-off-requires-explicit-moist",
                        f"{base_path}.parameters.moist",
                        "microphysics off leaves this real-source plan "
                        "without a moist carrier: the option's own row "
                        "carries moist=false -- the idealized reading -- "
                        "and nothing between plan review and the loader "
                        "rewrites it, so woof.ingest.real would refuse the "
                        "run before step 0 (`real initialization requires "
                        "cfg.moist=True`). Set moist=true "
                        f"{site} -- it allocates qv/qc/qr carrier fields "
                        "while microphysics stays off, and synthesizes no "
                        "analyzed clouds, so source-absent cloud mass stays "
                        "exact zero -- or select a microphysics scheme.",
                    )
                )

        for component_id, option_id in sorted(resolved_components.items()):
            component = components.get(component_id)
            options = component.get("options", {}) if isinstance(component, dict) else {}
            option = options.get(option_id) if isinstance(options, dict) else None
            constraints = option.get("constraints", {}) if isinstance(option, dict) else {}
            if not isinstance(constraints, dict):
                continue
            required_settings = constraints.get("required_settings", {})
            # ``required_settings_reasons`` mirrors ``requires_components_reasons``:
            # the breakage a pinned setting prevents, keyed by setting.  A
            # row without one is reported as the declaration it is (the
            # option's own parameters set the value and the registry
            # declares no other for it) rather than as a bare "requires".
            required_reasons = constraints.get("required_settings_reasons", {})
            if not isinstance(required_reasons, dict):
                required_reasons = {}
            if isinstance(required_settings, dict):
                for name, expected in required_settings.items():
                    if name not in settings or not _same_value(settings[name], expected):
                        observed = settings.get(name, "unset")
                        reason = required_reasons.get(name)
                        if not (isinstance(reason, str) and reason.strip()):
                            reason = (
                                f"the option's own parameters set {name}="
                                f"{expected!r} and the registry declares no "
                                f"other value of it for {option_id!r}")
                        errors.append(
                            _issue(
                                "component-required-setting",
                                f"{base_path}.components.{component_id}",
                                f"option {option_id!r} ({component_id}) requires "
                                f"{name}={expected!r}, got {observed!r}: "
                                f"{reason.strip()}. Set {name}={expected!r} "
                                f"for it, or select another {component_id} "
                                "option.",
                            )
                        )
            # A SETTING A SCHEME ADMITS MORE THAN ONE VALUE FOR.
            # ``required_settings`` can only say "exactly this", and a
            # scheme with two geometries then has to pick one of them and
            # refuse the other -- which is how plan review came to refuse
            # a six-level RUC column the loader admits and the kernel
            # runs.  This kind says the SET, is derived from the scheme's
            # own table (never hand-typed), and carries the reason the
            # gate law asks for beside it.
            admitted = constraints.get("admitted_setting_values", {})
            admitted_reasons = constraints.get(
                "admitted_setting_values_reasons", {})
            if not isinstance(admitted_reasons, dict):
                admitted_reasons = {}
            if isinstance(admitted, dict):
                for name, values in admitted.items():
                    if not isinstance(values, list):
                        continue
                    # An ABSENT setting resolves to the registry's own
                    # declared default, the way a conditional rule's does.
                    # Skipping it instead would let a plan that selects a
                    # scheme and never names the setting pass review and
                    # then be refused by validate_run_config, which is the
                    # two-door disagreement this kind was added to close --
                    # the num_soil_layers default is Noah's 4, and a RUC
                    # plan that says nothing is a RUC plan asking for 4.
                    observed = _effective_setting(
                        name, settings, parameter_specs)
                    if any(_same_value(observed, value) for value in values):
                        continue
                    reason = admitted_reasons.get(name)
                    detail = (f": {reason}"
                              if isinstance(reason, str) and reason else "")
                    errors.append(
                        _issue(
                            "component-admitted-setting",
                            f"{base_path}.components.{component_id}",
                            f"option {option_id!r} admits {name} in "
                            f"{values!r}, got {observed!r}{detail}",
                        )
                    )
            forbidden = constraints.get("forbidden_setting_values", {})
            if isinstance(forbidden, dict):
                for name, values in forbidden.items():
                    if not isinstance(values, list) or name not in settings:
                        continue
                    if any(_same_value(settings[name], value) for value in values):
                        errors.append(
                            _issue(
                                "component-forbidden-setting",
                                f"{base_path}.components.{component_id}",
                                f"option {option_id!r} forbids {name}={settings[name]!r}",
                            )
                        )
            required_components = constraints.get("requires_components", {})
            # A dependency row may carry the breakage it prevents, keyed by
            # the component it constrains.  "requires pbl in [off, mynn]"
            # says WHICH tuples are refused and nothing about WHAT breaks,
            # which is half a refusal: the reader cannot tell a transcribed
            # prohibition from a field contract, and the first reader who
            # decides it is the former deletes it.  Optional rather than
            # required here because most rows restate a setting the engine
            # refuses again with its own named message; where the registry
            # is the only voice, the row says why.
            dependency_reasons = constraints.get(
                "requires_components_reasons", {})
            if not isinstance(dependency_reasons, dict):
                dependency_reasons = {}
            if isinstance(required_components, dict):
                for required_id, allowed_options in required_components.items():
                    if (
                        not isinstance(allowed_options, list)
                        or resolved_components.get(required_id) not in allowed_options
                    ):
                        reason = dependency_reasons.get(required_id)
                        detail = (
                            f": {reason.strip().rstrip('.')}"
                            if isinstance(reason, str) and reason.strip()
                            else "")
                        observed = resolved_components.get(required_id, "unset")
                        errors.append(
                            _issue(
                                "component-dependency",
                                f"{base_path}.components.{component_id}",
                                f"option {option_id!r} ({component_id}) requires "
                                f"{required_id} in {allowed_options!r}, got "
                                f"{observed!r}{detail}. Select one of those "
                                f"{required_id} options with {option_id!r}, or a "
                                f"{component_id} option that admits {required_id} "
                                f"{observed!r}.",
                            )
                        )
            for rule in _conditional_refusals(constraints):
                if _conditional_refusal_fires(
                    rule, resolved_components, settings, parameter_specs,
                    normalized_context.get("source_id"),
                ):
                    # A rule carrying a ``sources`` clause gets its OWN code
                    # (audit R-005): a RunConfig carries no source identity,
                    # so it is the one refusal in this battery that the
                    # per-domain authority cannot mirror, and the agreement
                    # gate has to be able to say so by name rather than
                    # excusing the whole conditional-refusal class.
                    errors.append(
                        _issue(
                            "component-source-refusal"
                            if isinstance(rule.get("sources"), list)
                            else "component-conditional-refusal",
                            f"{base_path}.components.{component_id}",
                            f"option {option_id!r} is refused here: "
                            f"{conditional_refusal_sentence(rule)}",
                        )
                    )

        # CONSUMER ROWS.  Every consumer that will read this domain's
        # selected options during the run must have its row, and the
        # question is asked here -- the same function validate_run_config
        # asks it through -- so the two authorities cannot disagree about it.
        for name, spec in parameter_specs.items():
            if not isinstance(spec, Mapping) or not spec.get("requires_reader"):
                continue
            value = settings.get(name, spec.get("default"))
            if (not same_setting_value(value, spec.get("default"))
                    and not registry_knob_is_read(name, settings, registry)):
                errors.append(_issue("parameter-reader", f"{base_path}.parameters.{name}",
                                     str(spec.get("description", "the selected component has no reader for this setting"))))

        for component_id, option_id in sorted(resolved_components.items()):
            component = components.get(component_id)
            options = component.get("options", {}) if isinstance(component, dict) else {}
            option = options.get(option_id) if isinstance(options, dict) else None
            if not isinstance(option, dict):
                continue
            for gap in consumer_row_gaps_for_option(
                    component_id, option_id, option, settings):
                errors.append(
                    _issue(
                        "consumer-row-missing",
                        f"{base_path}.components.{component_id}",
                        gap,
                    )
                )

        resolved_domains.append(
            {
                "domain_id": domain_id,
                "template_id": template_id,
                "components": dict(sorted(resolved_components.items())),
                "settings": dict(sorted(settings.items())),
            }
        )

    # AUDIT R-058.  This compared template LABELS: two templates whose
    # component maps are byte-identical and differ only in a key the
    # loader accepts PER DOMAIN were refused for having different names,
    # and the message ("requires one uniform base template") named no
    # breakage at all.  The narrow constraint it stood in for is real and
    # is now stated where it lives -- ``parameters.<name>.per_domain``,
    # generated from the loader's own ``_DOMAIN_RUN_OVERRIDES`` -- and
    # checked on RESOLVED VALUES, so the refusal names the two values and
    # the one loader that cannot express them.
    shared_only = sorted(
        {
            name for name, spec in parameter_specs.items()
            if isinstance(spec, Mapping) and spec.get("per_domain") is False
        }
        | {
            key
            for component in components.values()
            if isinstance(component, Mapping)
            for key, per_domain in (
                component.get("per_domain_selectors", {}) or {}).items()
            if per_domain is False
        }
    )
    if isinstance(route, dict) and len(resolved_domains) > 1:
        for name in shared_only:
            observed: dict[str, object] = {}
            for resolved in resolved_domains:
                if name in resolved["settings"]:
                    observed.setdefault(
                        canonical_json(resolved["settings"][name]),
                        resolved["domain_id"])
            if len(observed) < 2:
                continue
            pairs = ", ".join(
                f"{domain_id} resolves {name}={value}"
                for value, domain_id in sorted(observed.items())
            )
            errors.append(
                _issue(
                    "nonuniform-shared-setting",
                    "domains",
                    f"{pairs}. woof carries ONE value of {name} for the "
                    "whole tree: the domain-tree loader's per-domain table "
                    "(woof/experiment.py _DOMAIN_RUN_OVERRIDES, which this "
                    "registry's parameters.per_domain rows are generated "
                    "from) does not carry it, so a [[domain]] table cannot "
                    "express a second one. Select one value for every "
                    "domain of the tree.",
                )
            )

    graph_constraints = (
        route.get("graph_setting_constraints", [])
        if isinstance(route, dict)
        else []
    )
    if isinstance(graph_constraints, list):
        child_ids = {edge["child_domain_id"] for edge in normalized_edges}
        root_ids = set(domain_ids) - child_ids
        for constraint in graph_constraints:
            if not isinstance(constraint, dict):
                continue
            setting_key = constraint.get("setting_key")
            scope = constraint.get("scope")
            reason = constraint.get("reason")
            # THE REASON IS PART OF THE ROW, not decoration.  These
            # constraints are topology questions no single-domain config
            # can answer, so this evaluator is the only voice that ever
            # explains them; without the reason the plan review printed
            # "non-root domain requires spec_exp=0.0, got 0.33" and left
            # the user to guess what a nonzero child value would do.  A
            # reasonless row is malformed, so a future one cannot ship
            # silent.
            if (not isinstance(setting_key, str)
                    or scope not in {"root", "non-root", "all"}
                    or not isinstance(reason, str)
                    or not reason.strip()):
                errors.append(
                    _issue(
                        "graph-setting-policy",
                        "registry.runner_routes",
                        "malformed graph setting constraint: it needs a "
                        "scope, a setting_key and a reason naming what "
                        "the constraint prevents",
                    )
                )
                continue
            expected = constraint.get("required_value")
            for domain_index, resolved in enumerate(resolved_domains):
                domain_id = resolved["domain_id"]
                selected = (
                    scope == "all"
                    or (scope == "root" and domain_id in root_ids)
                    or (scope == "non-root" and domain_id in child_ids)
                )
                if not selected:
                    continue
                observed = resolved["settings"].get(setting_key)
                if not _same_value(observed, expected):
                    errors.append(
                        _issue(
                            "graph-setting-constraint",
                            f"domains[{domain_index}].parameters.{setting_key}",
                            f"{scope} domain requires {setting_key}={expected!r}, got {observed!r}: {reason}",
                        )
                    )

    transition_policy_id = (
        route.get("transition_policy_id") if isinstance(route, dict) else None
    )
    if isinstance(transition_policy_id, str):
        transitions = selected_registry.get("transitions", {})
        transition_policy = (
            transitions.get(transition_policy_id)
            if isinstance(transitions, dict)
            else None
        )
        if not isinstance(transition_policy, dict):
            errors.append(
                _issue(
                    "transition-policy",
                    "registry.transitions",
                    f"runner route names missing transition policy {transition_policy_id!r}",
                )
            )
        else:
            component_id = transition_policy.get("component_id")
            resolved_by_id = {
                domain["domain_id"]: domain
                for domain in resolved_domains
                if isinstance(domain.get("domain_id"), str)
            }
            for edge_index, edge in enumerate(normalized_edges):
                parent = resolved_by_id.get(edge["parent_domain_id"])
                child = resolved_by_id.get(edge["child_domain_id"])
                if (
                    not isinstance(component_id, str)
                    or parent is None
                    or child is None
                ):
                    continue
                parent_option = parent["components"].get(component_id)
                child_option = child["components"].get(component_id)
                rule: object = None
                if parent_option == child_option:
                    same_option = transition_policy.get("same_option")
                    if isinstance(same_option, dict) and same_option.get("allowed") is True:
                        rule = same_option
                else:
                    cross_options = transition_policy.get("cross_options", [])
                    if isinstance(cross_options, list):
                        matches = [
                            candidate
                            for candidate in cross_options
                            if isinstance(candidate, dict)
                            and candidate.get("parent_option_id") == parent_option
                            and candidate.get("child_option_id") == child_option
                        ]
                        if len(matches) == 1:
                            rule = matches[0]
                edge_path = f"edges[{edge_index}]"
                if not isinstance(rule, dict):
                    errors.append(
                        _issue(
                            "unsupported-component-transition",
                            edge_path,
                            f"transition policy {transition_policy_id!r} does not admit {component_id} {parent_option!r}->{child_option!r}",
                        )
                    )
                    continue
                if parent_option != child_option:
                    # A published cross edge is only as real as its two
                    # endpoints' closures.  The registry used to publish ten
                    # mp=9 edges the nest-edge resolver refused at tree load,
                    # so plan review passed what the run start refused; the
                    # endpoint rows are generated from the resolver's own
                    # ported set, and an edge touching an unported endpoint
                    # is refused HERE with the resolver's reason.
                    for role, option_id in (("parent", parent_option),
                                            ("child", child_option)):
                        option = (components.get(component_id, {})
                                  .get("options", {}).get(option_id))
                        rows = (option.get(CONSUMER_ROWS_KEY, {})
                                if isinstance(option, dict) else {})
                        transition = rows.get("nest_transition") if isinstance(rows, dict) else None
                        if not isinstance(transition, dict):
                            continue
                        if transition.get("mixed_edge_ported") is not True:
                            errors.append(
                                _issue(
                                    "transition-unported-endpoint",
                                    edge_path,
                                    f"{role} {component_id} option "
                                    f"{option_id!r} has no ported mixed-edge "
                                    "closure, so the nest-edge resolver "
                                    "refuses this edge at tree load: "
                                    f"{transition.get('refusal')}.  Run the "
                                    "child on the parent's scheme (a "
                                    "same-scheme edge resolves before the "
                                    "closure is consulted), or choose a pair "
                                    "whose two endpoints both publish "
                                    "nest_transition.mixed_edge_ported=true",
                                )
                            )
                maturity = canonical_maturity(rule.get("maturity"), selected_registry)
                warning_maturities = _ladder_tier(selected_registry, "warn")
                if maturity in warning_maturities:
                    warnings.append(
                        _issue(
                            "transition-maturity",
                            edge_path,
                            f"{component_id} transition "
                            f"{parent_option!r}->{child_option!r} is "
                            f"{maturity!r}; review the per-species receipt",
                        )
                    )
                for role, resolved, key in (
                    ("parent", parent, "required_parent_settings"),
                    ("child", child, "required_child_settings"),
                ):
                    required = rule.get(key, {})
                    if not isinstance(required, dict):
                        continue
                    for name, expected in required.items():
                        observed = resolved["settings"].get(name)
                        if not _same_value(observed, expected):
                            errors.append(
                                _issue(
                                    "transition-required-setting",
                                    edge_path,
                                    f"{role} domain requires {name}={expected!r}, got {observed!r}",
                                )
                            )

    # A scheme whose consumers block declares a lateral-forcing dataset is
    # REPORTED here -- into ``install_state``, on the same domains the run
    # door refuses (woof.config.mp28_aerosol_lateral_forcing_precondition)
    # -- so plan review and the run cannot decide one tuple two ways.  Plan
    # review reports and does not raise, and it does not flip
    # ``launchable`` either: whether the 225 MB dataset is on THIS disk
    # says nothing about the plan, which is the same argument that keeps
    # the question out of validate_run_config (that battery is also the
    # namelist importer's, and an import emits a TOML and reads no
    # dataset).
    #
    # The condition is an EXTERNAL boundary: ``specified``, which every
    # real-data root sets (domain_wizard, hrrr_hierarchy_direct and
    # downscale all do).  Having a PARENT is not part of it and used to
    # be: nwfa and nifa are members of
    # woof.ingest.lateral_bc.COUPLED_SCALAR_STATE_FIELDS, so a nest edge
    # carries them from the parent whatever the parent's aerosol source
    # was, and refusing an idealized child named a breakage that path does
    # not have.  The row carries the dataset, the deliberate way out and
    # the sentence; nothing about the scheme is spelled in this module.
    for domain_index, resolved in enumerate(resolved_domains):
        if not bool(resolved["settings"].get("specified")):
            continue
        for component_id, option_id in resolved["components"].items():
            option = (components.get(component_id, {}).get("options", {})
                      .get(option_id))
            if not isinstance(option, dict):
                continue
            row = (option.get(CONSUMER_ROWS_KEY) or {}).get(
                "lateral_forcing_dataset")
            if not isinstance(row, dict):
                continue
            deliberate = row.get("deliberate_setting")
            if isinstance(deliberate, dict) and _same_value(
                    resolved["settings"].get(deliberate.get("name")),
                    deliberate.get("value")):
                continue
            resolution = resolve_lateral_forcing_dataset(
                row, resolved["settings"])
            if resolution["resolved"]:
                continue
            install_state.append(
                _issue(
                    "lateral-forcing-dataset",
                    # The path convention every per-domain issue in this
                    # module uses, so a reader -- and the authority-
                    # agreement gate -- can tell WHICH domain it is about.
                    f"domains[{domain_index}].components.{component_id}",
                    str(row.get("refusal", "")) + " Searched: "
                    + "; ".join(str(entry)
                                for entry in resolution["searched"]) + ".",
                )
            )

    asset_rows = []
    for key in sorted(asset_domains):
        requirement, domain_list, domain_indices = asset_domains[key]
        resolution = resolve_asset_requirement(requirement)
        asset_rows.append({
            "requirement": requirement,
            "domain_ids": sorted(domain_list),
            "resolution": resolution,
        })
        # ONE ROW PER DOMAIN, anchored at ``domains[<index>]``.  Every
        # other per-domain issue in this module is anchored there and a
        # reader takes the domain from the path; these two used to be
        # written as ``plan.domains[<grid id>,<grid id>]``, a shape that
        # matches no anchor and names no index, so a panel or a gate that
        # asked which domain an unresolved table set was about got no
        # answer for the one code that says a table is missing HERE.
        anchors = [f"domains[{domain_index}].asset_requirements."
                   f"{requirement.get('id')}"
                   for domain_index in sorted(domain_indices)]
        if resolution["resolved"]:
            continue
        if not resolution["missing"]:
            # The row declared nothing to look for (no ``assets``, no
            # ``relative_path``), so the ladder was never walked.  That is
            # not a satisfied requirement, it is an unanswerable one, and
            # reporting it satisfied is how an install short of a table
            # passed plan review and refused when the scheme loaded --
            # measured on the two RTE+RRTMGP rows (audit R-045).
            # tools/build_registry.py refuses to EMIT such a row; this is
            # the reader's half, so a registry from anywhere cannot buy a
            # vacuous pass.
            for anchor in anchors:
                errors.append(
                    _issue(
                        "asset-undeclared",
                        anchor,
                        "the option's required table set declares no files "
                        f"({resolution['origin']}), so plan review cannot "
                        "resolve it and would be reporting a check it never "
                        "made; the requirement must name its members or a "
                        "relative_path",
                    )
                )
            continue
        for anchor in anchors:
            install_state.append(
                _issue(
                    "asset-unresolved",
                    anchor,
                    "the option's required table set is not installed "
                    "HERE: "
                    + ", ".join(str(name) for name in resolution["missing"])
                    + " was not found under any declared root ("
                    + "; ".join(str(entry)
                                for entry in resolution["searched"])
                    + "). The plan is unchanged by this; install or stage "
                    "the set (`woof fetch-tables`) or point the declared "
                    "environment override at a byte-identical copy before "
                    "the run, which refuses by name when the scheme loads "
                    "its tables.",
                )
            )
    result["asset_requirements"] = asset_rows
    # ``install_state`` is deliberately not consulted: a machine short of
    # a table set has not made the plan wrong.
    result["launchable"] = not errors
    return result


def resolve_lateral_forcing_dataset(
        row: Mapping[str, object],
        settings: Mapping[str, object] | None = None) -> dict[str, object]:
    """Resolve a lateral-forcing dataset row through the module that owns it.

    THE reason this is not :func:`resolve_asset_requirement`.  The dataset
    a domain with external lateral boundaries (``specified``) needs is
    located at run time by an ingest module with its own ordered ladder, and a second ladder written in
    registry data resolved DIFFERENTLY: the ingest ladder has a
    working-directory rung (WRF's own ``constants_name`` rule) and accepts
    any filename through its single-file environment override, while a
    generic asset ladder looks for the canonical filename inside the
    override's parent directory and never looks at the working directory.
    Two ladders over one dataset is how plan review says LAUNCHABLE about a
    file the run cannot find, so the row names its resolver
    (``module:function``) and the answer comes from there.

    THE PLAN CARRIES NO MACHINE PATH, and this used to pretend it did.
    The row named a ``path_setting`` (``wif_climatology_path``) and this
    function read it out of the plan's settings -- but that name is not in
    the registry's ``parameters`` and no runner route lists it in
    ``allowed_parameter_keys`` or ``allowed_expert_setting_keys``, so
    ``_parameter_error`` refuses any plan that spells it and the branch
    could not run.  A plan is portable and a filesystem path is not; the
    operator-named path is a RunConfig field, answered by the run door,
    and here the resolver is asked with no argument, which is exactly what
    the environment overrides and the working-directory rung are for. The
    owner's own named refusal is reported as an unresolved answer carrying
    that sentence, because plan review reports rather than raises.

    ``settings`` is still accepted so the deliberate-setting test and this
    resolution read one signature; nothing in it names a path.
    """

    target = row.get("resolver")
    if not isinstance(target, str) or ":" not in target:
        fallback = resolve_asset_requirement(row)
        return {"resolved": bool(fallback["resolved"]),
                "searched": list(fallback["searched"])}
    module_name, _, attribute = target.partition(":")
    try:
        import importlib

        resolve = getattr(importlib.import_module(module_name), attribute)
        resolution = resolve(None)
    except Exception as error:            # noqa: BLE001 - reported, not raised
        return {"resolved": False,
                "searched": [f"{type(error).__name__}: {error}"]}
    return {
        "resolved": bool(getattr(resolution, "resolved", False)),
        "searched": [str(entry)
                     for entry in getattr(resolution, "candidates", ())],
    }


def resolve_asset_requirement(
        requirement: Mapping[str, object]) -> dict[str, object]:
    """Walk one asset requirement's declared ladder and say what answered.

    The ladder is data: ``requirement["resolution"]`` carries the same
    rungs the scheme's own loader walks -- a single-file environment
    override, a root environment override, the packaged root, and the
    root ``woof fetch-tables`` stages into -- and this function walks
    them with ``pathlib`` and :mod:`woof.data_assets`, which resolves a
    ``woof/data``-relative path to whichever distribution carries it.
    No physics module is imported and no table is read: existence and
    size decide, because the loaders re-verify SHA-256 before GPU setup
    and hashing 375 MB at plan review would buy nothing they do not
    already prove.

    Why this exists at all: ``validate_physics_plan`` used to COLLECT
    asset requirements and never resolve one, so an install missing a
    table passed plan review and refused inside the run -- the shape the
    gate law forbids.  A requirement that cannot resolve is now an error
    on the plan, naming the files and every root that was searched.
    """

    result: dict[str, object] = {
        "id": requirement.get("id"),
        "resolved": False,
        "root": None,
        "origin": None,
        "missing": [],
        "searched": [],
    }
    ladder = requirement.get("resolution")
    if not isinstance(ladder, dict):
        result["origin"] = "no-resolution-ladder"
        return result
    assets = requirement.get("assets")
    assets = assets if isinstance(assets, list) else []
    filenames = [
        str(asset.get("filename"))
        for asset in assets
        if isinstance(asset, dict) and asset.get("filename")
    ]
    sizes = {
        str(asset.get("filename")): asset.get("bytes")
        for asset in assets
        if isinstance(asset, dict) and asset.get("filename")
    }
    if not filenames:
        # A row that declares one file rather than a set (``relative_path``
        # names the file, the ladder names the directory it sits in).  A
        # requirement with neither is a requirement with nothing to check,
        # and resolving it vacuously would be a resolution that proves
        # nothing -- so it is named as such rather than reported resolved.
        relative_path = requirement.get("relative_path")
        if isinstance(relative_path, str) and relative_path:
            filenames = [relative_path.replace("\\", "/").rsplit("/", 1)[-1]]
        else:
            result["origin"] = "no-assets-declared"
            return result
    searched: list[str] = result["searched"]  # type: ignore[assignment]

    def _complete(root: Path) -> list[str]:
        missing = []
        for filename in filenames:
            member = root / filename
            expected = sizes.get(filename)
            try:
                ok = member.is_file() and (
                    expected is None or member.stat().st_size == expected)
            except OSError:
                ok = False
            if not ok:
                missing.append(filename)
        return missing

    candidates: list[tuple[str, Path]] = []
    path_env = ladder.get("path_environment_override")
    if isinstance(path_env, str) and os.environ.get(path_env):
        chosen = Path(os.environ[path_env])
        candidates.append(("$" + path_env, chosen.parent))
    root_env = ladder.get("root_environment_override")
    if isinstance(root_env, str) and os.environ.get(root_env):
        chosen_root = Path(os.environ[root_env])
        subdirectory = ladder.get("root_environment_subdirectory")
        if subdirectory:
            chosen_root = chosen_root / str(subdirectory)
        candidates.append(("$" + root_env, chosen_root))
    relative = ladder.get("data_relative")
    if isinstance(relative, str) and relative:
        try:
            from woof import data_assets

            candidates.append(("packaged", data_assets.data_path(relative)))
        except Exception as error:            # noqa: BLE001
            searched.append(f"packaged (unavailable: {error})")
    staged_env = ladder.get("staged_root_environment_override")
    if isinstance(staged_env, str) and os.environ.get(staged_env):
        candidates.append(
            ("$" + staged_env, Path(os.environ[staged_env]).expanduser()))
    staged = ladder.get("staged_root")
    if isinstance(staged, str) and staged:
        try:
            candidates.append(
                ("staged", Path(staged).expanduser()))
        except (RuntimeError, OSError):       # pragma: no cover - no home
            pass

    if (isinstance(root_env, str) and os.environ.get(root_env)
            and ladder.get("exclusive_root_environment_override")):
        candidates = [(origin, root) for origin, root in candidates
                      if origin == "$" + root_env]

    # WHICH ROOT'S MISSING LIST IS REPORTED.  The CLOSEST one -- the root
    # that holds the most of the set -- not the last one walked.  Reporting
    # the last was measured naming four files as missing while two of them
    # sat installed in the packaged root the ladder had already walked,
    # which sends an operator after 50 MB they already have.  An
    # instrument in a user-facing refusal must not misreport what it
    # measured; ties keep the earlier rung, which is the one with
    # precedence.
    closest: list[str] = list(filenames)
    closest_root: str | None = None
    for origin, root in candidates:
        searched.append(f"{origin}: {root}")
        missing = _complete(root)
        if not missing:
            result.update(resolved=True, root=str(root), origin=origin,
                          missing=[])
            return result
        if len(missing) < len(closest) or closest_root is None:
            closest, closest_root = missing, str(root)
    result["missing"] = closest
    result["closest_root"] = closest_root
    return result


def load_physics_plan(path: str | Path) -> object:
    """Load strict JSON for CLI validation without accepting NaN/Infinity."""

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )


__all__ = [
    "EXPERT_TEMPLATES_ANY_SOURCE",
    "expert_template_ids_for_source",
    "DEFAULT_TEMPLATE_ID",
    "INSTALL_STATE_CODES",
    "resolve_asset_requirement",
    "MORRISON_TEMPLATE_ID",
    "NSSL2_LEGACY_RRTMG_TEMPLATE_ID",
    "NSSL2_TEMPLATE_ID",
    "PLAN_SCHEMA",
    "REGISTRY_SCHEMA",
    "THOMPSON_KF_TEMPLATE_ID",
    "THOMPSON_TEMPLATE_ID",
    "VALIDATION_SCHEMA",
    "WSM6_TEMPLATE_ID",
    "canonical_json",
    "canonical_maturity",
    "canonical_template_id",
    "TEMPLATE_ID_ALIASES",
    "component_override_declaration",
    "canonical_sha256",
    "conditional_refusal_remedy",
    "conditional_refusal_sentence",
    "load_physics_plan",
    "physics_registry",
    "registry_sha256",
    "NO_OFF_VALUE",
    "REGISTRY_PHYSICS_IDENTITY_SCHEMA",
    "component_off_option",
    "recorded_registry_physics_parts",
    "registry_knob_is_read",
    "registry_knob_readers",
    "registry_off_template",
    "registry_physics_part_sources",
    "registry_physics_parts",
    "registry_physics_receipt",
    "registry_physics_sha256",
    "same_setting_value",
    "setting_off_value",
    "strip_registry_documentation",
    "validate_physics_plan",
]


def column_surface_contract(selector):
    """Resolve the record driver from the selected land-surface row."""
    for option in registry_view()["components"]["land_surface"]["options"].values():
        if option.get("selectors", {}).get("sf_surface_physics") == int(selector):
            return option.get("column_contract")
    return None


def validate_column_surface_settings(settings):
    """Check table-declared record couplings at both config and driver doors."""
    selector = _selection(settings, "sf_surface_physics") or 0
    contract = column_surface_contract(selector)
    if contract is None:
        return
    from woof.config import radiation_scheme_ids
    radiation = radiation_scheme_ids(settings)
    for name, row in contract["admission"].items():
        value = (radiation[0] if name == "ra_lw_physics" else
                 radiation[1] if name == "ra_sw_physics" else _selection(settings, name))
        if value not in row["values"]:
            raise ValueError(f"{name}={value}: {row['reason']}")
    if radiation[1] == 4 and _selection(settings, "ra_rrtmg_variant") != "rrtmg_legacy":
        raise ValueError("the selected column surface consumes visible and near-infrared "
                         "direct/diffuse radiation and returns four band albedos; "
                         "the RTE+RRTMGP adapter lacks these spectral carriers and cannot supply them. "
                         "Select ra_rrtmg_variant='rrtmg_legacy' or ra_sw_physics=1.")
