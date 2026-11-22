"""Data-only compatibility for supplied research-proposal catalogs v1/v2.

No archive member is extracted or executed. The converter preserves the source
records and records every executable-setting translation separately from prose.
"""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import PurePosixPath, PureWindowsPath
import stat
import zipfile


_TIERS = {"minimum": "lower", "preferred": "recommended", "large": "upper"}


def read_archive(raw: bytes, *, limit: int, object_pairs_hook) -> tuple[dict, dict]:
    from woof.case_catalog import CatalogError
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > 4096:
                raise CatalogError("The catalog ZIP exceeds the bounded member-count limit")
            seen = set()
            for row in entries:
                # Validate the original on-wire name before zipfile's Windows
                # separator/null normalization can hide a malformed member.
                name = row.orig_filename
                path = PurePosixPath(name)
                if (not name or "\x00" in name or "\\" in name or path.is_absolute() or PureWindowsPath(name).drive
                        or ".." in path.parts or any(":" in part for part in path.parts)
                        or stat.S_ISLNK(row.external_attr >> 16)):
                    raise CatalogError(f"Unsafe catalog ZIP member: {name!r}; paths must remain inside the archive")
                if name.casefold() in seen:
                    raise CatalogError(f"Duplicate catalog ZIP member: {name!r}")
                seen.add(name.casefold())
                if row.flag_bits & 1:
                    raise CatalogError("Encrypted catalog ZIP members are not supported")
            candidates = [row for row in entries if not row.is_dir() and PurePosixPath(row.filename).name == "catalog.json"]
            if len(candidates) != 1:
                raise CatalogError("Choose a catalog ZIP containing exactly one catalog.json")
            selected = candidates[0]
            if selected.file_size > limit:
                raise CatalogError("The expanded catalog.json exceeds the catalog size limit")
            with archive.open(selected) as stream:
                payload = stream.read(limit + 1)
            if len(payload) > limit:
                raise CatalogError("The expanded catalog.json exceeds the catalog size limit")
            document = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=object_pairs_hook)
            return document, {"catalog_member": selected.filename, "member_count": len(entries),
                              "expanded_bytes": sum(row.file_size for row in entries),
                              "selected_bytes_read": len(payload),
                              "members": [{"path": row.filename, "bytes": row.file_size} for row in entries],
                              "files_extracted": 0, "scripts_executed": False}
    except (zipfile.BadZipFile, UnicodeDecodeError, RuntimeError) as error:
        raise CatalogError(f"Cannot read this catalog ZIP: {error}") from error


def _cycle_ids(start: dict, by_id: dict) -> list:
    """Name the closed parent_id chain reached from ``start``."""
    walked = []
    cursor = start
    while cursor["id"] not in [row["id"] for row in walked]:
        walked.append(cursor)
        parent = cursor.get("parent_id")
        if parent is None:
            return []
        cursor = by_id[parent]
    return [row["id"] for row in walked[[row["id"] for row in walked].index(cursor["id"]):]]


def _resolve_nest_tree(domains: list, *, ident, tier_label: str) -> list:
    """Order a preset's domains parent before child from their declared ids.

    The catalog names each domain's nest parent with ``parent_id``. The id
    vocabulary and the order of the list are the catalog's own; only a tree a
    native tier cannot carry is refused, and each refusal names the domain rows
    that caused it.
    """
    from woof.case_catalog import CatalogError
    where = f"Case {ident}/{tier_label}"
    by_id: dict = {}
    for position, domain in enumerate(domains):
        declared = domain.get("id")
        if not isinstance(declared, str) or not declared:
            raise CatalogError(f"{where}: domain row {position + 1} of {len(domains)} declares no id, so no nest can name it as its parent. Give every domain row an id.")
        if declared in by_id:
            raise CatalogError(f"{where}: domain id {declared!r} is declared twice, so a parent_id naming it picks out two domains. Give every domain row its own id.")
        by_id[declared] = domain
    roots, children = [], {}
    for domain in domains:
        parent = domain.get("parent_id")
        if parent is None:
            roots.append(domain)
            continue
        if parent not in by_id:
            raise CatalogError(f"{where}: domain {domain['id']!r} names nest parent {parent!r}, which this preset does not declare, so the nest has nothing to sit inside. Declare that parent domain, or point the child at one of {', '.join(repr(name) for name in sorted(by_id))}.")
        children.setdefault(parent, []).append(domain)
    if not roots:
        closed = _cycle_ids(domains[0], by_id)
        raise CatalogError(f"{where}: domains {', '.join(repr(name) for name in closed)} name each other as nest parents in a closed parent_id chain, so no domain is the outermost one. Leave the outermost domain's parent_id absent or null.")
    if len(roots) > 1:
        raise CatalogError(f"{where}: domains {', '.join(repr(row['id']) for row in roots)} each declare no nest parent, and one tier carries one outermost domain. Give every domain except the outermost one a parent_id, or move the extra root into its own source_domain_recipes entry.")
    ordered, cursor = [], roots[0]
    while cursor is not None:
        ordered.append(cursor)
        kin = children.get(cursor["id"], [])
        if len(kin) > 1:
            raise CatalogError(f"{where}: domains {', '.join(repr(row['id']) for row in kin)} all nest inside {cursor['id']!r}, and a tier's geometry is one nest ladder (root_dx_km plus nest_ratios), which no single chain spells. Declare one recipe per branch under source_domain_recipes, or build the sibling nests with woof domain.")
        cursor = kin[0] if kin else None
    if len(ordered) < len(domains):
        reached = {row["id"] for row in ordered}
        stranded = [row for row in domains if row["id"] not in reached]
        closed = _cycle_ids(stranded[0], by_id)
        raise CatalogError(f"{where}: domains {', '.join(repr(name) for name in closed)} name each other as nest parents in a closed parent_id chain, so they never reach the outermost domain {roots[0]['id']!r}. Point one of them at a domain that reaches the outermost one.")
    return ordered


def _convert_proposal(document: dict, archive_report: dict | None = None, *, version: int) -> dict:
    from woof.case_catalog import (CatalogError, SCHEMA, _native_contract,
                                   validate_native_overrides)
    from woof.physics_compat import THOMPSON_RTE_RRTMGP_PROFILE_ID
    source_schema = f"arwen.case-catalog/v{version}"
    if document.get("schema") != source_schema:
        raise CatalogError(f"Unsupported imported catalog schema {document.get('schema')!r}")
    if not str(document.get("catalog_version", "")).startswith(f"{version}."):
        raise CatalogError(f"This converter supports research-proposal catalog version {version}.x")
    original_cases = document.get("cases")
    if not isinstance(original_cases, list) or not original_cases:
        raise CatalogError("The imported catalog has no cases list")
    _, shared_keys, domain_keys = _native_contract()
    initializers = document.get("initialization_sources", {})
    profiles = document.get("physics_profiles", {})
    converted = []
    mapped_counts = {"cases": 0, "source_options": 0, "fixed_diff_opt_2": 0}
    issues = []
    for original in original_cases:
        ident = original.get("id")
        case_issues, transforms = [], []
        initialization = original.get("initialization", {})
        default_source = initialization.get("default_source_id")
        options = []
        alternatives = (list(initialization.get("source_options", {}).values()) if version == 2
                        else initialization.get("alternatives", []))
        for alternative in alternatives:
            source_id = alternative["source_id"]
            source_info = initializers.get(source_id, {})
            native_source = source_info.get("engine_source_id") or source_id
            schedules = {}
            for old_tier, tier in _TIERS.items():
                schedule = alternative["schedules"][old_tier]
                hours = schedule["total_integration_hours"]
                if isinstance(hours, bool) or not isinstance(hours, (int, float)) or int(hours) != hours:
                    raise CatalogError(f"Case {ident}, {source_id}/{old_tier}: native integration duration must be a whole number of hours")
                schedules[tier] = {"cycle_utc": schedule["start_utc"], "run_hours": int(hours),
                                   "forecast_start_hour": 0, "metadata": deepcopy(schedule)}
                if version == 2:
                    schedules[tier]["cadence_hours"] = schedule["boundary_interval_hours"]
                    schedules[tier]["metadata"]["source_status"] = deepcopy(alternative.get("status_by_tier", {}).get(old_tier, {}))
            options.append({"id": source_id, "source": native_source,
                            "cycle_utc": schedules["recommended"]["cycle_utc"],
                            "cadence_hours": source_info.get("requested_boundary_cadence_hours", source_info.get("boundary_interval_hours", source_info.get("time_cadence_hours", 1))),
                            "tier_schedules": schedules,
                            "label": source_info.get("label", source_id),
                            "notes": "The selected tier determines its actual cycle and duration. Archive/date eligibility is not confirmation that source files exist.",
                            "metadata": {"original_source_id": source_id, "source_definition": deepcopy(source_info),
                                         "proposal": deepcopy(alternative)}})
        default_option = next((row for row in options if row["id"] == default_source), None)
        if default_option is None:
            raise CatalogError(f"Case {ident}: the recommended initialization source is absent")
        tiers = {}
        for old_tier, tier in _TIERS.items():
            preset = original["presets"][old_tier]
            domains = preset["domains"]
            if not domains:
                raise CatalogError(f"Case {ident}/{old_tier} has no domain intents")
            # The catalog owns its id vocabulary and the order of this list; the
            # nest tree is the one its parent_id fields declare.
            ordered = _resolve_nest_tree(domains, ident=ident, tier_label=old_tier)
            intents, selectors = [], []
            for index, domain in enumerate(ordered):
                intents.append({"grid_id": index + 1, "center_lat": domain["center_lat"],
                                "center_lon": domain["center_lon"], "dx_km": domain["dx_km"],
                                "width_km": domain["width_km"], "height_km": domain["height_km"]})
                chosen = deepcopy(domain.get("selectors", {}))
                if "diff_opt" in chosen:
                    value = chosen.pop("diff_opt")
                    if isinstance(value, bool) or value not in (1, 2):
                        case_issues.append(f"{old_tier}/{domain['id']}: diff_opt={value!r} is unsupported; select model-coordinate horizontal diffusion 1 or metric stress/scalar diffusion 2")
                    elif value == 2:
                        mapped_counts["fixed_diff_opt_2"] += 1
                        transforms.append({"tier": tier, "grid_id": index + 1, "field": "diff_opt", "value": 2,
                                           "binding": "The existing woof.namelist_import contract maps diff_opt=2 to WOOF-ArWen's native mixing form; km_opt remains explicit."})
                    else:
                        chosen["diff_opt"] = value
                        transforms.append({"tier": tier, "grid_id": index + 1,
                                           "field": "diff_opt", "value": value,
                                           "binding": "Model-coordinate horizontal diffusion is preserved; km_opt remains explicit."})
                selectors.append(chosen)
            common = {key: value for key, value in selectors[0].items()
                      if all(row.get(key) == value for row in selectors)}
            shared = {}
            for key, value in common.items():
                if key in shared_keys:
                    shared[key] = value
                else:
                    case_issues.append(f"{old_tier}: native shared setting {key} is not supported")
            per_domain = []
            for index, (domain, chosen) in enumerate(zip(ordered, selectors)):
                values = {}
                for key, value in chosen.items():
                    if key in common:
                        continue
                    if key in domain_keys:
                        values[key] = value
                    else:
                        case_issues.append(f"{old_tier}/{domain['id']}: per-domain {key} is not supported by the native configuration scope")
                if values:
                    per_domain.append({"grid_id": index + 1, "settings": values})
            profile_id = original.get("physics", {}).get("recommended_profile_id")
            profile = profiles.get(profile_id, {})
            if "soil_layers_required" in profile:
                shared["num_soil_layers"] = profile["soil_layers_required"]
            overrides = {"shared": shared, "domains": per_domain}
            try:
                validate_native_overrides(overrides)
            except ValueError as error:
                case_issues.append(f"{old_tier}: {error}")
                # Preserve the full original selectors and block creation.
                # An unsupported choice must not poison browsing other cases.
                overrides = {}
            spacing = {domain["id"]: intent["dx_km"] for domain, intent in zip(ordered, intents)}
            ratios = []
            for domain in ordered[1:]:
                ratio = spacing[domain["parent_id"]] / spacing[domain["id"]]
                if int(ratio) != ratio:
                    raise CatalogError(f"Case {ident}: grid spacings do not form integer native nest ratios")
                ratios.append(int(ratio))
            tiers[tier] = {"root_dx_km": intents[0]["dx_km"], "nest_ratios": ratios,
                           "domain_intents": intents, "run_hours": default_option["tier_schedules"][tier]["run_hours"],
                           "nz": preset["vertical_grid"]["levels_intent"],
                           "native_overrides": overrides,
                           "notes": preset.get("purpose", "") + " Native projection, aligned dimensions, source coverage and memory still require admission."}
        profile_id = original.get("physics", {}).get("recommended_profile_id")
        metadata = {"imported_case": deepcopy(original), "imported_physics_profile": deepcopy(profiles.get(profile_id, {})),
                    "conversion": {"source_schema": source_schema, "version": version, "issues": case_issues,
                                   "translations": transforms,
                                   "native_profile_role": "Starting profile only; the imported explicit selectors determine final science.",
                                   "unapplied_advisory_intents": ["timestep starting guesses", "radiation-update intent", "vertical model-top/near-surface targets", "optional LES and sensitivity experiments", "output/verification proposals"]}}
        if version == 2:
            # Reuse the same typed domain/physics conversion for each explicitly
            # supplied source geometry. No source is silently given the common
            # 12/3/1 layout when its proposal instead declares a 3/1 layout.
            variants = {option.get("domain_recipe") for option in alternatives} - {None, "standard_12_3_1"}
            for variant in sorted(variants):
                variant_case = deepcopy(original)
                for old_tier in _TIERS:
                    preset = variant_case["presets"][old_tier]
                    declared_variant = preset.get("source_domain_recipes", {}).get(variant)
                    if not isinstance(declared_variant, dict) or not declared_variant.get("domains"):
                        raise CatalogError(f"Case {ident}/{old_tier}: missing source domain recipe {variant!r}")
                    preset["domains"] = deepcopy(declared_variant["domains"])
                for option in variant_case["initialization"]["source_options"].values():
                    option["domain_recipe"] = "standard_12_3_1"
                variant_document = dict(document, cases=[variant_case])
                variant_result = _convert_proposal(variant_document, version=version)["cases"][0]
                for option, original_option in zip(options, alternatives):
                    if original_option.get("domain_recipe") == variant:
                        option["tier_geometry"] = deepcopy(variant_result["tiers"])
                        option["metadata"]["conversion"] = deepcopy(variant_result["metadata"]["conversion"])
                        option["metadata"]["conversion"]["source_domain_recipe"] = variant
        converted.append({"id": ident, "title": original["name"], "event_kind": original["category"],
                          "summary": original.get("selection_rationale", ""),
                          "source_options": options, "recommended_source_option": default_source,
                          # A default, not a pin: woof/case_catalog.py binds this
                          # field when the caller names no profile, so it follows
                          # the product's radiation arm (RTE+RRTMGP on both streams,
                          # cumulus off) rather than the legacy twin it replaced,
                          # which differs in the radiation engine and nothing else.
                          # A proposal that means a specific engine names it per
                          # case instead of inheriting one here.
                          "tiers": tiers, "physics_profile": THOMPSON_RTE_RRTMGP_PROFILE_ID,
                          "recommendations": [{"topic": "physics", "data": deepcopy(original.get("physics", {}))},
                                              {"topic": "verification", "data": deepcopy(original.get("verification", {}))}],
                          "tags": deepcopy(original.get("tags", [])), "metadata": metadata})
        mapped_counts["cases"] += 1
        mapped_counts["source_options"] += len(options)
        issues.extend({"case_id": ident, "message": message} for message in case_issues)
    return {"schema": SCHEMA,
            "catalog": {"id": "arwen-historical-case-proposals", "title": document.get("title", "Imported case catalog"),
                        "version": document["catalog_version"],
                        "description": "Imported research proposals. Case identity, source availability and simulation skill are not certified by importing this catalog.",
                        "provenance": [{"source_schema": source_schema, "generated_utc_date": document.get("generated_utc_date"),
                                        "metadata": deepcopy(document.get("metadata", {}))}],
                        "metadata": {"import_mapping": {"converter": f"research-proposal-v{version}", "version": version,
                                                        "counts": mapped_counts, "issues": issues,
                                                        "archive": archive_report},
                                     "source_registry": deepcopy(document.get("source_registry", {})),
                                     "initialization_sources": deepcopy(initializers),
                                     "physics_profiles": deepcopy(profiles),
                                     "sensitivity_experiments": deepcopy(document.get("sensitivity_experiments", {}))}},
            "cases": converted}


def convert_proposal_v1(document: dict, archive_report: dict | None = None) -> dict:
    return _convert_proposal(document, archive_report, version=1)


def convert_proposal_v2(document: dict, archive_report: dict | None = None) -> dict:
    return _convert_proposal(document, archive_report, version=2)
