# SPDX-License-Identifier: Apache-2.0
"""Whether this release has a derivative for every kernel kind a recorded plan runs, decided when the plan is sealed.

A kind is covered when it ran in an example plan whose window passed the release's check (data/derivative-kinds.v1.json,
made by tools/derivative_kinds.py from the pinned example plans), when the window gives it a rule by name
(rust/hexad/src/runner/wrules.rs, Ctx::kernel_rule and its helpers), when it writes no float word (a validator, or a
CuPy kernel whose result type is bool or an integer, as the window decides by its name), when the window holds it by
scheme, or when a door row holds it by name (``derivative_holds`` in data/ad-doors.v1.json: the window lists such kinds
in its refuse list). Any other kind refuses the recording: the window would have no derivative for it, or would keep it
as identity over the words it writes.
THE BREAKAGE THIS PREVENTS: a 12 km plan recorded with 0.6.0 sealed and replayed exactly, and the release's own check
then failed on it (whole-state dot 0.228 over 15 minutes; the gradient promised 0.71 to 0.77 of the real change)
because the forecast ran cupy_getitem_mask, a kind no example plan runs and no rule covers. Nothing said so until a
window ran (2026-11-18). What this cannot see: a covered kind whose arguments its derivative refuses (the
route's words in route.py choose those forms).

A checked kind is also refused when a launch hands it two pointer arguments at one address in a pair no example plan
ran it with (aliased_pairs). THE BREAKAGE THIS PREVENTS: with the dycore's short array lifetimes the last RK stage's
recover_edges_f32 wrote flux_u over its own input ru_avg_p (and recover_interfaces_f32, recover_cells_f32 likewise);
their reverse rules add into the input's cotangent and then clear the output's, so on one address the input's share is
lost. The plan sealed and replayed exactly, the tangent matched the real model, and the whole-state dot read 0.239
after one step (2026-11-18). The derivative route records the earlier lifetimes (route.py,
WOOF_HEX_MEM_ACOUSTIC).
"""
from __future__ import annotations

import collections
import json
import re
from functools import lru_cache
from pathlib import Path

from . import env

KINDS_FILE = env.DATA / "derivative-kinds.v1.json"

#: kinds the window gives a rule by name (wrules.rs: kernel_rule's match arms, the integer-result, module-constant,
#: constant-table, regional-driving, mesh-scaling, specified-zone and scrub helpers, image constants and its INACTIVE
#: list). A kind added to the window's rules is added here in the same change.
WINDOW_RULES = frozenset((
    "rrfs_post_transport_copy", "rrfs_glue_copy_plan", "halo_copy_bytes", "halo_copy_words_2d",
    "ruc_land_tile_preflight", "rrfs_radiation_unpack", "accumulate", "rrfs_init_microphysics_top", "ruc_tbq_init",
    "mpas_init_tables", "mixing_scaling_f32", "regional_lbc_derive_v841", "regional_lbc_rho_edge_v841",
    "regional_lbc_tendency_v841", "regional_lbc_state_at_v841", "regional_speczone_tend_cell_v841",
    "regional_speczone_tend_edge_v841", "regional_speczone_u_ru_v841", "regional_zero_speczone_w_v841",
    "regional_reset_speczone_values_v841", "zero_speczone_lagged_diabatic_theta_v842_f32",
    "solve_region_scrub_columns_v841", "tempo_warp_sizes", "cupy_fill", "cupy_maximum__float_float32_float32",
    "cupy_copy_where__float32_bool_float32", "cupy_true_divide__float32_float_float32",
    "cupy_multiply__float32_float_float32", "fct_scale", "fct_minmax_source", "upper_cam_f32", "upper_rayleigh_f32",
    # INACTIVE (bool or integer words, validators, diagnostics, column-order permutations)
    "cupy_less__float32_float32_bool", "cupy_not_equal__int32_int_bool", "cupy_cub_any_pass1", "cupy_cub_any_pass2",
    "hex_transport_species_words", "hex_transport_species_levels", "validate_density_v841",
    "validate_finite_array_v841_f32", "validate_recovered_v841_f32", "w_filter_geometry_v841_f32", "smooth_helicity",
    "split_helicity_pool", "vertex_helicity_gather_pool", "update_diag_pool", "mpas_output_diagnostics_v842",
    "oracle_diags_sfcdiags_ruclsm", "tempo_radar", "tempo_order_hist", "tempo_order_scan", "tempo_order_scatter",
    "compute_diag_cols", "reset_diag_pool", "solve_region_validate_v841", "cupy_cub_all_pass1", "cupy_cub_all_pass2",
    "mpas_mcica_seeds_column",
    # CuPy's plain boolean reductions (any() and all() off the CUB path): bool in, bool out. Every limited-area kernel
    # set (storm-h, storm-h-ws, kset-d12) gives cupy_any the class "inactive", which wrules.rs runs as identity both
    # ways. THE BREAKAGE THIS PREVENTS: a 30 km regional recording (16,933 cells, 2024-05-21 14Z) was refused at
    # sealing for "cupy_any (3 launches)": the short-wave KISS pressure check that deferred_checks_v842 reduces once a
    # radiation call.
    "cupy_any", "cupy_all",
    # The external-tendency slot (woof/hex/ad/external_tendency.py): tend += ext, wrules.rs external_tendency_rule
    # (R::Axpy, c = 1). Launched only when a caller gives an external tendency.
    "external_tendency_f32",
))
#: kind prefixes and kinds the window holds by scheme (wrules.rs: the held:<scheme> fallback)
HELD_PREFIXES = ("tempo_", "mynnsfc_", "ugwp_oro_", "oracle_")
HELD_KINDS = frozenset(("rrfs_mynnedmf_run_l16", "rrfs_mynnedmf_run_l4"))
_CUPY = re.compile(r"^cupy_[a-z0-9_]+?__([a-z0-9_]+)$")


@lru_cache(maxsize=1)
def checked_kinds() -> dict:
    return json.loads(KINDS_FILE.read_text())["kinds"]


@lru_cache(maxsize=1)
def held_by_name() -> dict[str, str]:
    """{kind: the door row and hold that names it}, from every row's derivative_holds."""
    try:
        from woof.hex import ad_doors
        rows = ad_doors.load_table()["rows"]
    except Exception:  # noqa: BLE001 - a recorder without the doors' table holds nothing by name
        return {}
    held: dict[str, str] = {}
    for row_name, row in rows.items():
        for hold, spec in (row.get("derivative_holds") or {}).items():
            for kind in spec.get("kinds", ()):
                held.setdefault(kind, f"{row_name}: {hold}")
    return held


def _no_float_word(kind: str) -> bool:
    if kind.startswith("validate_"):
        return True
    if kind.startswith("cupy_copy__"):
        types = kind[len("cupy_copy__"):].split("_")
        return len(types) == 2 and not types[1].startswith("float")
    found = _CUPY.match(kind)
    if not found:
        return False
    out = found.group(1).rsplit("_", 1)[-1]
    return out in ("bool", "int") or bool(re.fullmatch(r"u?int\d+", out))


def why_covered(kind: str) -> str | None:
    """How the release's derivative covers KIND, or None when it does not."""
    if kind in checked_kinds():
        return "checked"
    if kind in WINDOW_RULES:
        return "rule"
    if _no_float_word(kind):
        return "no-float-word"
    if kind in HELD_KINDS or kind.startswith(HELD_PREFIXES):
        return "held-by-scheme"
    if kind in held_by_name():
        return "held-by-name"
    return None


#: Window rules a runner binary carries only from the release that wrote them, by the window kernel each launches (the
#: runner looks the kernel up by this name, so a runner that has the rule holds the name). THE BREAKAGE THIS PREVENTS:
#: on 18 Nov two sealed plans (30 km regional 1 h, 16,933 cells; 6 km 36 min, 10,736 cells) were recorded by the 0.6.2
#: recorder, whose coverage counts these kinds as rules, and their windows ran the 0.6.1 runner of the venv they were
#: overlaid on, which has neither rule: every window stopped at its first damping launch with "no derivative row and no
#: rule", which reads as a missing row, and the plans were taken for uncheckable until the runner was read.
RUNNER_RULE_KERNELS = {"upper_cam_f32": b"win_upper_cam", "upper_rayleigh_f32": b"win_upper_rayleigh"}


def runner_lacks(kinds, runner: Path | None) -> list[str]:
    """The kinds among KINDS whose window rule the runner binary at RUNNER does not carry (empty when RUNNER is None)."""
    want = sorted(k for k in kinds if k in RUNNER_RULE_KERNELS)
    if runner is None or not want:
        return []
    data = Path(runner).read_bytes()
    return [k for k in want if RUNNER_RULE_KERNELS[k] not in data]


def kinds_of(plan_json: Path) -> collections.Counter:
    plan = json.loads(Path(plan_json).read_text())
    return collections.Counter(op["function"] for ops in plan["blocks"].values() for op in ops
                               if op.get("kind") == "kernel")


def _pairs(op: dict) -> set[tuple[int, int]]:
    seen: dict[tuple, list[int]] = {}
    for i, arg in enumerate(op.get("arguments") or ()):
        target = arg.get("target") if isinstance(arg, dict) else None
        if isinstance(target, dict) and "allocation" in target:
            seen.setdefault((target["allocation"], target.get("offset", 0)), []).append(i)
    return {(a[x], a[y]) for a in seen.values() for x in range(len(a)) for y in range(x + 1, len(a))}


def aliased_pairs(plan_json: Path) -> dict[str, dict]:
    """{kind: {"pairs": [[i, j], ...], "launches": n}}: checked kinds launched with pointer arguments i and j at one
    address where no example plan ran the kind so (the table's aliased_args). Their derivative was checked only in
    the forms the example plans ran; a reverse rule that clears an output's cotangent after adding into an input's
    loses the input's share when the two are one word."""
    table = checked_kinds()
    plan = json.loads(Path(plan_json).read_text())
    out: dict[str, dict] = {}
    for op in (op for ops in plan["blocks"].values() for op in ops if op.get("kind") == "kernel"):
        kind = op.get("function")
        if kind not in table:
            continue
        new = _pairs(op) - {tuple(p) for p in table[kind].get("aliased_args", ())}
        if new:
            row = out.setdefault(kind, {"pairs": set(), "launches": 0})
            row["pairs"] |= new
            row["launches"] += 1
    return {k: {"pairs": sorted(map(list, v["pairs"])), "launches": v["launches"]} for k, v in sorted(out.items())}


def check(plan_json: Path, runner: Path | None = None) -> dict:
    """{"kinds", "by", "held_by_name", "missing": {kind: launches}, "aliased": {kind: {pairs, launches}}, "table",
    "rule_kinds", "runner_lacks"} for a sealed plan; RUNNER (the recording's runner) is read for the rules it carries."""
    count = kinds_of(plan_json)
    by: collections.Counter = collections.Counter()
    missing: dict[str, int] = {}
    named: list[str] = []
    rules: list[str] = []
    for kind, launches in sorted(count.items()):
        how = why_covered(kind)
        if how is None:
            missing[kind] = launches
        else:
            by[how] += 1
            if how == "rule":
                rules.append(kind)
            if how == "held-by-name":
                named.append(f"{kind} ({held_by_name()[kind]})")
    return {"kinds": len(count), "by": dict(sorted(by.items())), "held_by_name": named, "missing": missing,
            "aliased": aliased_pairs(plan_json),
            "table": {"file": KINDS_FILE.name, "sha256": env.sha256_file(KINDS_FILE)},
            "rule_kinds": rules, "runner_lacks": runner_lacks(rules, runner)}


def refuses(result: dict) -> bool:
    """Whether a sealed plan's coverage (check) refuses it."""
    return bool(result["missing"] or result.get("aliased"))


def refusal(result: dict) -> str:
    """The one-line refusal for a plan with kinds the release has no derivative for, or runs in an unchecked form."""
    parts = []
    if result["missing"]:
        named = ", ".join(f"{k} ({n} launches)" for k, n in sorted(result["missing"].items(), key=lambda x: -x[1]))
        parts.append(f"the recorded forecast runs {len(result['missing'])} kernel kind(s) this release has no "
                     f"derivative for: {named}")
    if result.get("aliased"):
        named = ", ".join(f"{k} (arguments {' and '.join('/'.join(map(str, p)) for p in v['pairs'])} at one "
                          f"address, {v['launches']} launches)" for k, v in result["aliased"].items())
        parts.append(f"the recorded forecast hands {len(result['aliased'])} kernel kind(s) two arguments at one "
                     f"address where no example plan did, a form their derivative was not checked in: {named}")
    return ("; ".join(parts) + ". A window on this plan would fail its check or keep those kinds as identity, so "
            "nothing was sealed (plan.json is kept as plan.refused.json, derivative-coverage.json names every kind). "
            "Record on --route derivative, or leave out the forecast option that runs them")
