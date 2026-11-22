#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Point the any-mesh door rows at one published adjoint set ("LATEST").

The adjoint program publishes one agreed set of files per id (the merged runner, the kernel sets, the scheme bundles,
the TEMPO images and the exact command of the best run per mesh size) with a MANIFEST.json. This tool turns that
manifest into door table data, so the public doors run the same operator the program measures:

* the template row ``hrrrv5-any`` (the fast composition: the manifest's best recipes) and ``hrrrv5-any-exact`` (the
  exact composition: the manifest's default recipe on the mesh-generic kernel set);
* per suite, the one-card window classes by mesh size (each class is one manifest recipe's sizes and environment);
* an asset root whose ``latest`` entry is the published tree, plus the two small row files (refuse list, parameters).

    ad_doors_from_latest.py --manifest <latest>/MANIFEST.json --table tree/src/hexcore/data/ad-doors.v1.json
    ad_doors_from_latest.py --manifest ... --asset-root <asset-root> --latest <latest>

Nothing here is a code path per mesh: a new size class is one more entry in CLASSES (a recipe name and a cell bound).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# runner environment variables and flags that name files of the published tree -> the row's asset key
ENV_ASSET = {"AD_SCHEMES": "bundle", "AD_DOT_TEMPO_SERIAL_CUBIN": "tempo_serial_cubin",
             "AD_TEMPO_BLOCK_CUBIN": "tempo_block_cubin", "AD_WIN_TEMPO_TLF_CUBIN": "tempo_tlf_cubin",
             "AD_TAPE_UNREAD": "tape_unread"}
ARG_ASSET = {"kernels": "kernels", "cubins": "cubins", "schemes": "schemes", "tempo-cubin": "tempo_cubin"}
# flags the door itself supplies (the plan, the window length, the output, the direction and the objective)
DOOR_ARGS = {"config", "steps", "out", "bstore", "seeds", "zero", "direction", "stable-dir", "x-scale", "y-scale",
             "refuse"}
# environment of the measuring harness, not of the operator
SKIP_ENV = {"AD_DIGEST_PROF", "CUDA_DEVICE_ORDER", "CUDA_HOME", "CUDA_VERSION",
            # the measuring deck of a recipe (the published 3km carries them): AD_DOT_SELF reseeds the reverse with the
            # tangent result, so on the door's objective window it would replace the objective's seed (the door sets
            # it only on its own second window); AD_DOT_ALLOC splits the dot per allocation; the reference payloads
            # name the recipe's own package, never the plan a door user brings
            "AD_DOT_SELF", "AD_DOT_ALLOC", "AD_BINDER_REFERENCE_PAYLOADS",
            # the switches of a proof leg (NaN into the left-out memory, and the run it is compared with)
            "AD_DOT_DEAD_POISON", "AD_DOT_DEAD_CONTROL"}
# The strict self dot's left-out memory (fold 1's opt-in): AD_DOT_DEAD names scratch memory the self dot gives zero
# weight, AD_DOT_DEAD_PROOF the recipe's own proof run. They belong to the self-dot window only and name files of the
# published tree, so they become the class's "self_dot_dead" (paths under the asset root), never runner environment.
# THE BREAKAGE THIS PREVENTS: kept as environment, the 120 km class handed every door window the literal
# "{L}/lists/dead120-tail-s3.json" (no {L} is filled in an environment value), and the door's strict dot had no way
# to use the list or prove it (box-d 2026-11-11).
DOT_DEAD_ENV = {"AD_DOT_DEAD": "list", "AD_DOT_DEAD_PROOF": "recipe_proof"}

# one-card window classes: (largest mesh in cells, fast recipe, exact recipe sizes come from the same class)
# "memory" is the admission rule of the class: the door asks for the card memory the window was MEASURED to hold
# (fixed_mib + mib_per_column_level x cells x levels x margin + margin_mib). The runner grows into free memory, so a
# peak read on a big free card over-asks on a smaller one: the 480 km rule asked 35,860 MiB and refused every door on
# a 32 GB RTX 5090 whose windows held at most 24,360 MiB there (2026-11-11). Its fixed part is now read on
# that smallest card. Other peaks read on 2026-11-09 from door receipts (doors/runs/*/
# receipt.json, card_peak_mib) and the published set's gates (fold gate c-480-best summary.json).
CLASSES = [
    {"max_cells": 12000, "recipe": "480-best",
     "why": "the sizes of the 480 km recipe; the tape sizes itself and the TEMPO arena grows into free memory",
     "memory": {"fixed_mib": 16500, "mib_per_column_level": 0.052, "margin": 1.0, "margin_mib": 2000,
                "source": "the card memory the window holds on the SMALLEST card measured, plus a stated margin of "
                          "2,000 MiB: x1.2562 x 59 levels, 3 steps, on a 32,607 MiB RTX 5090 held 19,406 (tangent) "
                          "to 24,360 MiB (sensitivity), 2026-11-11 (the door receipts' card_peak_mib; the card's "
                          "peak 24,360). The fixed part is that peak less the per-column-level cost (24,360 - 0.052 "
                          "x 2,562 x 59 = 16,500). The per-column-level cost is the slope of the free 80 GB card "
                          "peaks (x1.2562 35,510; 4,905-cell zoom globe 41,520; x1.10242 58,954 MiB, all x 59 "
                          "levels): on a big card the runner grows into free memory, about 12,000 MiB more at "
                          "x1.2562, and that growth is taken as fixed. It asks 26,360 / 33,548 / 49,922 MiB for "
                          "those three meshes; only x1.2562 is measured on a small card"}},
    {"max_cells": 50000, "recipe": "120",
     "why": "the sizes of the 120 km one-card recipe (a whole 96 GB card)",
     # the host memory the window holds: the door refuses before any download on a host with less available
     "need_host_mib": 81920,
     "need_host_source": "host memory of the 120 km window, 80 GiB: through the four commands on the default row (3 steps, "
                         "one 96 GB card, 2026-11-12) the largest host peak was about 75,000 MiB (adjoint-test, with its "
                         "self-dot and proof windows); one measured point and about 9 per cent over it",
     # THE BREAKAGE: the rule asked 78,838 MiB for x1.40962 x 59 levels and the exact suite's self-dot window held
     # 84,864 MiB (a window admitted on less runs in a memory layout nobody measured); the fast suite held at most
     # 71,156. Each suite's rule is now its measured door peak plus 2,000 MiB (box-d d120-e, 2026-11-11, receipt.json
     # card_peak_mib in each 120 km door receipt)
     "memory": {"fixed_mib": 17915, "mib_per_column_level": 0.02203, "margin": 1.0, "margin_mib": 2000,
                "source": "fast suite door peak on a free 96 GB RTX PRO 6000 plus 2,000 MiB: x1.40962 x 59 levels, 3 "
                          "steps, plan pk120e4; adjoint-test 71,156 (self-dot window), sensitivity 69,888, tangent "
                          "62,186 MiB (2026-11-11, receipt.json card_peak_mib). Fixed part = 71,156 less "
                          "the per-column-level cost (0.02203 x 40,962 x 59); the slope is not measured at a second "
                          "mesh size"},
     "exact_memory": {"fixed_mib": 31623, "mib_per_column_level": 0.02203, "margin": 1.0, "margin_mib": 2000,
                      "source": "exact suite door peak plus 2,000 MiB: x1.40962 x 59 levels, 3 steps, the recorded plan; "
                                "adjoint-test 84,864 MiB (self-dot and proof windows; objective window 83,730), a rented card "
                                "d120-e 2026-11-11, receipt.json card_peak_mib"}},
]
# a multi-card class joins the table the moment the published set carries its recipe (a valid window exists)
MULTI = {"max_cells": 200000, "recipe": "60",
         "why": "the 60 km recipe: the physics columns sharded over the recipe's cards and the step tape streamed",
         # THE BREAKAGE: this class had no memory rule, so a 60 km row fell back to the suite's provisional fit
         # (20,000 + 0.029 x cells x levels) and asked 300,334 MiB for x1.163842 x 59 levels: every plain 60 km door
         # was refused as needing more than the whole 96 GB card, though the recipe's window holds 82,958 MiB on one
         # card (doors6003, 2026-11-11). The ask is now the recipe's measured peak plus 2,000 MiB.
         # the fast suite's door holds more than the recipe's window (its objective window keeps the Taylor states):
         # with 84,958 MiB the door was admitted on a card it then filled 3,702 MiB past its ask (d60-p0001-fast2)
         "memory": {"fixed_mib": 90660, "mib_per_column_level": 0.0, "margin": 1.0, "margin_mib": 0,
                    "source": "the door's own measured fast-suite card peak plus 2,000 MiB: x1.163842 x 59 levels, 3 steps, the recorded plan; adjoint-test fast 88,660 MiB (2026-11-11, receipt.json card_peak_mib; admitted on the old rule's 84,958, from the recipe's 82,958, it held 3,702 MiB more); one measured point, so no per-column-level cost"},         # the exact suite holds more card memory than the fast one: its own measured door peak plus 2,000 MiB
         "exact_memory": {"fixed_mib": 90642, "mib_per_column_level": 0.0, "margin": 1.0, "margin_mib": 0,
                          "source": "exact suite door peak plus 2,000 MiB: x1.163842 x 59 levels, 3 steps, the recorded plan; adjoint-test 88,642 MiB (2026-11-11, receipt.json card_peak_mib; admitted on the fast rule's 84,958 it held 3,684 MiB more); one measured point, so no per-column-level cost"}}
# a limited-area class joins the table the moment the published set carries its recipe. It is picked by the mesh's
# kind (a grid whose bdyMaskCell marks a boundary zone), never by size: a 38,278-cell storm box and a 40,962-cell
# globe need different kernel sets (the limited-area gather tables), physics decks and refuse lists. It is WHOLE:
# the class carries its recipe's every file, flag and environment, and its own refuse list (the 3 km recipe has none:
# radiation and land run live), instead of adding a few sizes to the global template.
LAM = {"mesh_kind": "limited-area", "max_cells": 50000, "recipe": "3km", "exact_recipe": "3km-exact",
       "why": "the 3 km storm box recipe of development work: kernel set with the limited-area gather tables, the relax "
              "zone differentiated, the lateral-boundary driving data held (zero tangent)",
       "memory": {"fixed_mib": 93000, "mib_per_column_level": 0.0, "margin": 1.0,
                  "source": "ONE measured point: the 38,278-cell 3 km box x 59 levels, 4-step window, card peak "
                            "90.8 GB (the storm3km rows, recipe 3km needs.card_mib 93000); one WHOLE card"}}
# The limited-area class for meshes above the storm box's (and for any limited-area mesh on a card smaller than the
# storm-box class needs): the 3 km recipe's operator words, with its sizes taken from the mesh and the card. It is
# derived from the whole 3 km class (fast and exact alike), so it carries that recipe's every file, flag and variable
# except these. THE BREAKAGE THIS PREVENTS: the 135,318-cell Warn-on-Forecast-sized 3 km box was refused at the door
# ("above this suite's largest one-card window class (50,000 cells)") with --cards 0,1 as well, though the runner
# records it and its out-of-core budget holds any window on one card (wofs-1, 2026-11-13); and storm-h's column-solve
# row would take 40*ncells*(128*(nlev+1)+64) bytes, about 43 GB, in that box's reverse scratch.
LAM_BOUNDED = {
    "max_cells": 200000, "suffix": "-bounded", "kset": "storm-h-ws",
    "why": ("limited-area meshes of 50,001 to 200,000 cells, and any limited-area mesh on a card with less memory than "
            "the whole 3 km class needs: the 3 km recipe's operator on kernel set storm-h-ws (its column-solve row's "
            "scratch bound to the size it uses), sized to the mesh and to the card it is admitted on, the step tape "
            "streamed through host memory and every byte past the card budget held in host memory"),
    # the class's own words (one per whole recipe it is made from). They name its own window and kernel set: the
    # whole class's words describe the 38,278-cell box on one whole card and kernel set storm-h, which this class
    # does not run (door-row check F2, 2026-11-13)
    "what": {
        "3km": ("3 km limited-area mesh of up to 200,000 cells on one card, or two with --cards A,B (the MYNN-EDMF "
                "and TEMPO columns split over them), the fast composition of the 3 km recipe (MYNN-EDMF bundle B9 "
                "with the guarded boundary-layer images laid over it (the turbulence closure, plume and returned "
                "height differentiated under the weak-shear guard; the species outputs of the four number tendencies "
                "held), TEMPO block reverse that keeps no tape, fp32 TEMPO tangent; kernel set storm-h-ws; radiation "
                "and land live, NO refuse list; --local-topology 1 --mesh-constants 1). The check it makes is the "
                "strict self dot (AD_DOT_SELF=1). Labelled approximations: the FCT limiter runs live on windows up to"
                " fct_live_max_s and holds its coefficients past that; the species outputs of the boundary layer's "
                "four number tendencies held; lateral-boundary driving data held (zero tangent), the relax zone "
                "itself is differentiated; --boundary-layer taped runs B9's own held boundary layer instead"),
        "3km-exact": ("3 km limited-area mesh of up to 200,000 cells on one card, or two with --cards A,B (the MYNN-EDMF "
                      "and TEMPO columns split over them), the EXACT composition of the 3 km recipe (bundle gen-A, taped "
                      "TEMPO, fp64 TEMPO tangent, kernel set storm-h-ws, radiation and land live, NO refuse list). "
                      "Labelled approximations: FCT limiter coefficients held; lateral-boundary driving data held (zero "
                      "tangent), the relax zone itself is differentiated; the MYNN-EDMF boundary layer, its plume "
                      "included, is differentiated here (bundle gen-A)"),
    },
    # the drag words after the sized words (as the whole classes say them, for a mesh of this class)
    "what_after": {
        "3km": ("The mountain-drag closure is live in this class (bundle gen-B9, as in the whole-card 3 km class): "
                "below about 3.6 km grid spacing (ls_taper under 0.01 in ugwp_oro_v842.py) the drag scheme skips "
                "its mesoscale drag and blocking entirely, and the closure (the stress limiter and the Richardson "
                "number) belongs to that part, so on a mesh that fine it has nothing to act on. On a coarser "
                "limited-area mesh it acts, and its derivative can grow without bound after about an hour, as on "
                "the 120 km globe."),
        "3km-exact": ("The mountain-drag closure is differentiated (bundle gen-A). Below about 3.6 km grid spacing "
                      "(ls_taper under 0.01 in ugwp_oro_v842.py) the drag scheme skips its mesoscale drag and blocking "
                      "entirely, and the closure (the stress limiter and the Richardson number) belongs to that part, so "
                      "on a mesh that fine it has nothing to act on; on a coarser limited-area mesh it acts, and its "
                      "derivative can grow without bound after about an hour, as on the 120 km globe."),
    },
    "what_sized": ("SIZED TO THE MESH AND THE CARD: kernel set storm-h-ws binds the column-solve row's scratch to the "
                   "size it uses (the hand-written row, whose words differ from the per-cell graph row of the whole 3 km "
                   "class by rounding); reverse scratch from the mesh's size; the step tape streamed through host "
                   "memory; a card budget of the admitted card's free memory less 3,072 MiB and a host budget of 0.8 "
                   "of the available host memory, with every byte past the card budget in host memory"),
    # the 3 km recipe's sizes that become the mesh's and the card's (flag -> new value); the rest is the recipe's
    "args_set": {"--scratch-gib": "{scratch_gib}", "--tempo-arena-gib": "4"},
    "args_add": ["--spill-cold", "1", "--spill-mode", "migrate", "--vram-budget-gb", "{vram_gb}",
                 "--host-budget-gb", "{host_gb}"],
    "env_set": {"AD_TAPE_STREAM": "1", "AD_TEMPO_ARENA_GIB": "4"},
    "card_words": {
        "{scratch_gib}": {"from": "mesh", "bytes_per_column_level": 1576, "unit_bytes": 1073741824, "add": 1, "min": 4,
                          "source": "the largest reverse-scratch row once the column solve is bound, "
                                    "transport_standard_finish_regional_v841: 3,315,856,560 bytes at the 38,278-cell "
                                    "storm box x 55 levels (door window.json arena_use_by_function, 2026-11-11), "
                                    "1,576 bytes a column level; plus 1 GiB. The runner grows the arena past it if a "
                                    "launch asks (scratch_fit)"},
        "{vram_gb}": {"from": "card_free_mib", "less_mib": 3072, "unit_mib": 1024, "min": 8,
                      "source": "the admitted card's free memory less 3,072 MiB for the CUDA context and the module "
                                "images, which the runner's budget does not count"},
        "{host_gb}": {"from": "host_available_mib", "fraction": 0.8, "unit_mib": 1024, "min": 16,
                      "source": "0.8 of the host memory available at admission, the runner's own rule for a rank "
                                "with no --host-budget-gb (ranks.rs)"}},
    # --cards A,B: the MYNN-EDMF and TEMPO columns of each launch split over the named cards (ordinals within them)
    "cards_max": 2, "need_free_mib_shard": 8192,
    "shard_env": {"AD_MYNN_SHARD_DEVICES": "{ordinals}", "AD_TEMPO_SHARD_DEVICES": "{ordinals}"},
    "memory": {"fixed_mib": 16384, "mib_per_column_level": 0.0, "margin": 1.0, "margin_mib": 0,
               "source": "a floor, not a measured peak: the class's card words hold the runner inside the admitted "
                         "card's free memory less 3,072 MiB and put every byte past it in host memory (the runner's "
                         "out-of-core budget: where bytes live changes, no word does); below 16,384 MiB free the "
                         "door does not try"},
}
LAM_BOUNDED_ABOVE = ("a limited-area mesh above 200,000 cells needs the mesh split across cards with one rank per card; "
                     "no such limited-area window recipe is in the published set yet")


def bounded_lam_class(whole: dict) -> dict:
    """The bounded limited-area class from a whole limited-area class (fast or exact): same recipe, same files except
    the kernel set, with the mesh's and the card's sizes (LAM_BOUNDED). Table data; a size word is a {token} the door
    fills (rows.fill_mesh_words when the row is made, rows.fill_card_words at each admission)."""
    spec = LAM_BOUNDED
    args = list(whole["window_args"])
    for flag, value in spec["args_set"].items():
        if flag in args:
            args[args.index(flag) + 1] = value
        else:
            args += [flag, value]
    args += [a for a in spec["args_add"]]
    assets = dict(whole["assets"])
    assets["kernels"] = f"latest/kset/{spec['kset']}/kernels.json"
    assets["cubins"] = f"latest/kset/{spec['kset']}/cubins"
    out = {k: v for k, v in whole.items() if k not in ("need_host_mib", "need_host_mib_by_door", "need_host_source")}
    if whole["from_recipe"] not in spec["what"]:
        raise SystemExit(f"bounded limited-area class: no words for recipe {whole['from_recipe']!r} in LAM_BOUNDED['what']")
    out.update({
        "max_cells": spec["max_cells"], "from_recipe": whole["from_recipe"] + spec["suffix"], "why": spec["why"],
        "window_args": args, "window_env": dict(sorted({**whole["window_env"], **spec["env_set"]}.items())),
        "assets": assets, "memory": dict(spec["memory"]), "card_words": json.loads(json.dumps(spec["card_words"])),
        "cards": 1, "cards_max": spec["cards_max"], "need_free_mib_shard": spec["need_free_mib_shard"],
        "shard_env": dict(spec["shard_env"]),
        "what": f"{spec['what'][whole['from_recipe']]}. {spec['what_sized']}. "
                f"{spec['what_after'][whole['from_recipe']]}"})
    return out


def add_bounded_lam(table: dict) -> dict:
    """Append the bounded limited-area class after each suite's whole limited-area class (hrrrv5, hrrrv5-exact), once."""
    out = json.loads(json.dumps(table))
    for name in ("hrrrv5", "hrrrv5-exact"):
        suite = out["suites"][name]
        classes = suite["window_classes"]
        whole = [c for c in classes if c.get("mesh_kind") == "limited-area" and c.get("whole")
                 and not str(c.get("from_recipe", "")).endswith(LAM_BOUNDED["suffix"])]
        classes[:] = [c for c in classes if not str(c.get("from_recipe", "")).endswith(LAM_BOUNDED["suffix"])]
        if whole:
            classes.insert(classes.index(whole[0]) + 1, bounded_lam_class(whole[0]))
            suite["window_class_above_limited_area"] = LAM_BOUNDED_ABOVE
    return out


LAM_ABOVE = ("a limited-area mesh needs the published limited-area window recipe (3km), which this published set "
             "does not carry")
ABOVE_ONE = ("a mesh above 50,000 cells does not fit one card's tape: it needs the multi-card window recipe (60 km: "
             "column shards over 4 cards and the streamed tape), which is not in this table until one such window is "
             "valid by its own flags")
ABOVE_ALL = ("a mesh above 200,000 cells (finer than about 60 km on the globe) needs the mesh split across cards "
             "with one rank per card; no such window recipe is in the published set yet")
EXACT_RECIPE = "480"
# the mesh-generic kernel sets: kset-d9 = kset-d7 + project_edges and moist_edge_coefficients safe on
# limited-area and rank meshes (LATEST 20261109T2350Z-7904c5c3 moved every fast recipe to it; d7 leaves the tree after
# one more id). Every later kset-dN of the dycore rows is built the same way (kset-d10 = kset-d9 with unread branch records
# bound null, LATEST 20261110T0124Z-f82af038). The rows take the highest kset-dN (N >= 7) the published recipes name,
# so a door never names a set the tree dropped and needs no edit when dycore publishes the next one.
GENERIC_KSET = "kset-d7"
_GENERIC = re.compile(r"/kset/kset-d(\d+)/")


def generic_kset(manifest: dict) -> str:
    named = {str(v) for r in (manifest.get("recipes") or {}).values() if isinstance(r, dict)
             for f, v in (r.get("args") or []) if f == "kernels"}
    numbers = [int(m.group(1)) for v in named for m in [_GENERIC.search(v)] if m and int(m.group(1)) >= 7]
    return f"kset-d{max(numbers)}" if numbers else GENERIC_KSET


def _split(recipe: dict) -> dict:
    """One manifest recipe -> {assets, env, args, refuse} with paths relative to the asset root."""
    rel = lambda v: v.replace("{L}/", "latest/")
    assets, env, args, refuse, dead = {}, {}, [], "", {}
    for key, value in recipe["env"].items():
        if key in SKIP_ENV:
            continue
        if key in DOT_DEAD_ENV:
            dead[DOT_DEAD_ENV[key]] = rel(value)
            continue
        if key in ENV_ASSET:
            assets[ENV_ASSET[key]] = rel(value)
        else:
            env[key] = value
    for flag, value in recipe["args"]:
        if flag == "refuse":
            refuse = value
        elif flag in DOOR_ARGS:
            continue
        elif flag in ARG_ASSET:
            assets[ARG_ASSET[flag]] = rel(value)
        else:
            args.append((flag, value))
    return {"assets": assets, "env": env, "args": args, "refuse": refuse, "dot_dead": dead}


def _flat(pairs) -> list[str]:
    out: list[str] = []
    for flag, value in pairs:
        out += ["--" + flag, str(value)]
    return out


def _common(parts: list[dict]) -> dict:
    """What every class recipe agrees on (the template row) and what each adds (its class)."""
    # a file that every class names is in the template row (the first class's file is the default; a class overrides)
    assets = {k: v for k, v in parts[0]["assets"].items() if all(k in p["assets"] for p in parts)}
    env ={k: v for k, v in parts[0]["env"].items() if all(p["env"].get(k) == v for p in parts)}
    args = [a for a in parts[0]["args"] if all(a in p["args"] for p in parts)]
    return {"assets": assets, "env": env, "args": args}


def _generic(assets: dict, kset: str = GENERIC_KSET) -> dict:
    """The mesh-generic kernel set in place of a mesh-pinned one."""
    out = dict(assets)
    out["kernels"] = f"latest/kset/{kset}/kernels.json"
    out["cubins"] = f"latest/kset/{kset}/cubins"
    return out


def _lam_class(recipe: dict, spec: dict, what: str, exact: bool) -> dict:
    """A WHOLE limited-area class from one manifest recipe: every file, flag and variable, and its own refuse list."""
    part = _split(recipe)
    env_keys = {v: k for k, v in ENV_ASSET.items()}
    assets = dict(part["assets"])
    assets["refuse"] = f"rows/hrrrv5-any/refuse-{what}.txt"
    args = _flat(part["args"])
    if "--local-topology" not in args:
        args += ["--local-topology", "1"]
    return {"mesh_kind": spec["mesh_kind"], "whole": True, "max_cells": spec["max_cells"], "cards": 1,
            "from_recipe": what, "why": spec["why"], "window_args": args,
            "window_env": dict(sorted(part["env"].items())), "assets": assets,
            "asset_args": {"--" + flag: key for flag, key in ARG_ASSET.items() if key in assets},
            "asset_env": {env_keys[key]: key for key in sorted(assets) if key in env_keys},
            "refuse_list": part["refuse"], "exact": exact, "memory": spec.get("memory"),
            **({"self_dot_dead": {**part["dot_dead"], "from_recipe": what}} if part.get("dot_dead") else {}),
            # the recipe's own description: a whole class runs that recipe, not the template's composition
            "what": recipe.get("what") or f"the published recipe {what}"}


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def fast_words(assets: dict, args: list[str], refused: bool) -> str:
    """The fast template's words, from the files and flags it runs: what is held, named in the opening clause, and
    the labelled simplifications. THE BREAKAGE THIS PREVENTS: the words were a constant, so a composition that also
    held the boundary layer's closure, the drag closure or the limiter opened "every physics derivative live except
    radiation" (0.6.0 checks, 16 Nov 2026; tests/test_ad_row_holds_words.py reads every row against its fields)."""
    bundle = Path(assets.get("bundle") or "").name
    b9 = "B9" in bundle and not bundle.startswith("phys-")
    held = (["radiation"] if refused else []) + (
        ["the boundary layer's turbulence closure and MYNN-EDMF plume"] if b9 else []) + (
        ["the mountain-drag closure"] if "gfrz" in bundle else []) + (
        ["the FCT limiter coefficients"] if "--fct-frozen" in args
        and args[args.index("--fct-frozen") + 1] == "1" else [])
    simple = (["the MYNN-EDMF boundary layer's turbulence closure, mass-flux plume and returned height are frozen at "
               "the trajectory (bundle B9)"] if b9 else []) + (
        [f"the mountain-drag closure (GSL drag) is held at the trajectory (bundle {bundle})"] if "gfrz" in bundle
        else []) + [
        "the TEMPO tangent runs in " + ("double precision" if "ti2" in Path(assets.get("tempo_tlf_cubin") or "").name
                                        else "float32")] + (
        ["the TEMPO adjoint is the block reverse (recomputed, not taped)"] if assets.get("tempo_block_cubin") else [])
    return ("RRFS hrrrv5 suite with every physics derivative live"
            + (f" except {_and(held)}, which are held (below)" if held else "")
            + "; labelled simplifications in the derivative only (the forward is the recorded forecast): "
            + ", ".join(simple))


# the derivative images a person may swap into a row or class after the published set (the drag-held bundle, the
# float64 TEMPO tangent); a re-run on the same published set keeps them with the words that describe them
CARRIED_IMAGES = ("bundle", "tempo_tlf_cubin", "tempo_block_cubin")


def carry_choices(table: dict, rows: dict, suites: dict, lid: str) -> None:
    """A re-run on the published set the table already follows keeps what was chosen after it: each template row's
    and window class's words and swapped derivative images, the suites' words, and a whole limited-area class as the
    table took it. THE BREAKAGE THIS PREVENTS: a re-run wrote back the words the 0.6.0 checks had removed (a recipe's
    development description, "B9 holds the MYNN plume" on rows that differentiate it) and the images they described
    (the fast template's gen-B9 and float32 TEMPO tangent over the table's drag-held bundle and float64 tangent)."""
    if (table["rows"].get("hrrrv5-any") or {}).get("latest_id") != lid:
        return
    env_of = {v: k for k, v in ENV_ASSET.items()}
    for name in ("hrrrv5-any", "hrrrv5-any-exact"):
        was, new = table["rows"].get(name) or {}, rows[name]
        if was.get("what"):
            new["what"] = was["what"]
        for k in CARRIED_IMAGES:
            if k in (was.get("assets") or {}) and k in new["assets"]:
                new["assets"][k] = was["assets"][k]
    for sname in ("hrrrv5", "hrrrv5-exact"):
        old_suite = (table.get("suites") or {}).get(sname) or {}
        if old_suite.get("what"):
            suites[sname]["what"] = old_suite["what"]
        classes = suites[sname]["window_classes"]
        for i, new in enumerate(classes):
            same = [c for c in old_suite.get("window_classes") or [] if c.get("from_recipe") == new.get("from_recipe")
                    and c.get("mesh_kind") == new.get("mesh_kind")]
            if not same:
                continue
            if new.get("whole"):
                classes[i] = json.loads(json.dumps(same[0]))
                continue
            if same[0].get("what"):
                new["what"] = same[0]["what"]
            for k in CARRIED_IMAGES:
                if k in (same[0].get("assets") or {}):
                    new["assets"][k] = same[0]["assets"][k]
                    if k in env_of:
                        new["asset_env"][env_of[k]] = k


def build(manifest: dict, table: dict) -> dict:
    recipes = manifest["recipes"]
    lid = manifest["id"]
    spec_list = list(CLASSES)
    multi = recipes.get(MULTI["recipe"]) or {}
    if multi.get("args") and multi.get("env") is not None:
        spec_list.append({**MULTI, "cards": int((multi.get("needs") or {}).get("cards") or 1)})
    above = ABOVE_ALL if len(spec_list) > len(CLASSES) else ABOVE_ONE
    fast = [_split(recipes[c["recipe"]]) for c in spec_list]
    exact = _split(recipes[EXACT_RECIPE])
    refuse = fast[0]["refuse"]
    if any(p["refuse"] != refuse for p in fast + [exact]):
        raise SystemExit("the recipes do not share one refuse list; the row carries one list")
    base = _common(fast)
    old = table["rows"]["hrrrv5-any"]
    keep = {k: old[k] for k in ("suite", "levels", "default_steps", "state_copy", "holds_physics", "record_env",
                                "record_env_why", "kset_mesh", "template_only", "boundary_layer",
                                "fct_live_max_s", "fct_live_max_what") if k in old}

    def row(what: str, common: dict, extra_args: list[str], class_keys: set[str]) -> dict:
        assets = _generic(common["assets"], generic_kset(manifest))
        assets.update({"refuse": "rows/hrrrv5-any/refuse.txt", "params": "rows/hrrrv5-any/params.json"})
        env_keys = {v: k for k, v in ENV_ASSET.items()}
        return {
            "what": what, "mesh": None, "dt_s": None, "cells": None, "edges": None, "need_free_mib": 1, **keep,
            "assets": assets,
            "asset_args": {"--" + flag: key for flag, key in ARG_ASSET.items()},
            "asset_env": {env_keys[key]: key for key in sorted(assets) if key in env_keys},
            "window_args": _flat(common["args"]) + extra_args,
            "window_env": dict(sorted(common["env"].items())),
            "doors": ["tangent", "adjoint-test", "sensitivity"],
            "doors_why": ("param-grad needs a parameter-carrier bundle of this composition; the one that exists for the published set (rows/hrrrv5-any/bundle-params) is the fast composition\'s (plume held): rows made with --suite hrrrv5 run param-grad"),
            "image_archs": ["sm_120"],
            "latest_id": lid,
            "runner_needs": f"the window runner this wheel carries (woof/hex/libexec/ad-binder), with the files of the published set {lid}",
        }

    local = [] if ("local-topology", "1") in base["args"] else ["--local-topology", "1"]
    fast_assets = _generic(base["assets"], generic_kset(manifest))
    fast_args = _flat(base["args"]) + local
    frozen = "--fct-frozen" in fast_args and fast_args[fast_args.index("--fct-frozen") + 1] == "1"
    tail = (("; radiation is held (its launches are refused by name)" if refuse else "")
            + "; two set-up kernels of the first step that read no model state run as identity (mynnsfc_tables, "
            "w_filter_geometry_v841_f32)" + ("; FCT limiter coefficients frozen from the trajectory" if frozen else "")
            + "; the kernel set builds its topology tables per launch from the run's own mesh (--local-topology 1), "
            "so one kernel set serves every mesh")
    rows = dict(table["rows"])
    rows["hrrrv5-any"] = row(
        f"template of the rows made on demand (--mesh), fast composition of the published set {lid}: "
        + fast_words(fast_assets, fast_args, bool(refuse)) + tail, base, local, {k for p in fast for k in p["assets"]})
    # param-grad on the fast template: its parameter-carrier bundle, pin and settings are carried over from the table
    # being rewritten (they belong to the published set the bundle was composed on, not to the manifest)
    if "param_bundle" in old and old.get("latest_id") == lid:
        fast_row = rows["hrrrv5-any"]
        fast_row["doors"] = fast_row["doors"] + ["param-grad"]
        fast_row.pop("doors_why")
        fast_row["assets"]["bundle_params"] = old["assets"]["bundle_params"]
        fast_row["param_names"], fast_row["param_bundle"] = old["param_names"], old["param_bundle"]
    exact_common = {"assets": exact["assets"], "env": exact["env"],
                    "args": [a for a in exact["args"] if a in base["args"]]}
    rows["hrrrv5-any-exact"] = row(
        f"template of the rows made on demand (--mesh --suite hrrrv5-exact), exact composition of the published "
        f"set {lid}: the same suite without the fast row's plume and TEMPO simplifications in the derivative "
        "(MYNN-EDMF plume differentiated, float64 TEMPO tangent, taped TEMPO adjoint); slower and larger than the "
        "fast row. Held in this composition as in the fast row" + tail, exact_common,
        ["--local-topology", "1"], {"tape_unread"})

    def classes(parts: list[dict], common: dict) -> list[dict]:
        out = []
        for spec, part in zip(spec_list, parts):
            out.append({
                "max_cells": spec["max_cells"], "from_recipe": spec["recipe"], "why": spec["why"],
                "cards": spec.get("cards", 1),
                **{k: spec[k] for k in ("need_host_mib", "need_host_mib_by_door", "need_host_source") if spec.get(k)},
                "window_args": _flat([a for a in part["args"] if a not in common["args"]]),
                "window_env": {k: v for k, v in sorted(part["env"].items()) if common["env"].get(k) != v},
                "assets": {k: v for k, v in sorted(part["assets"].items())
                           if common["assets"].get(k) != v and k not in ("kernels", "cubins")},
                "memory": spec.get("memory"),
                # the runner variables of the files only this class carries (the template names its own)
                "asset_env": {k: v for k, v in sorted(ENV_ASSET.items())
                              if v in part["assets"] and common["assets"].get(v) != part["assets"][v]},
                **({"self_dot_dead": {**part["dot_dead"], "from_recipe": spec["recipe"]}}
                   if part.get("dot_dead") else {}),
            })
        return out

    fast_classes = classes(fast, base)
    # the exact row takes each class's sizes (flags and arena environment), never its fast images
    size_env = ("AD_TEMPO_ARENA_GIB", "AD_P3_ARENA_X", "AD_P3_TILE_MAX", "AD_MYNN_SHARD_DEVICES",
                "AD_TEMPO_SHARD_DEVICES", "AD_TAPE_STREAM", "AD_ANCHORS_AHEAD")
    exact_classes = [{**c, "assets": {k: v for k, v in c["assets"].items() if k == "tape_unread"},
                      "asset_env": {k: v for k, v in c["asset_env"].items() if v == "tape_unread"},
                      "window_env": {k: v for k, v in c["window_env"].items() if k in size_env},
                      "from_recipe": f"{EXACT_RECIPE} with the sizes of {c['from_recipe']}",
                      # the exact composition holds more card memory than the fast one: its own measured rule
                      "memory": spec.get("exact_memory") or c["memory"]}
                     for spec, c in zip(spec_list, fast_classes)]
    lam = recipes.get(LAM["recipe"]) or {}
    if lam.get("args") and lam.get("env") is not None:
        fast_classes.append(_lam_class(lam, LAM, LAM["recipe"], False))
        lam_exact = recipes.get(LAM["exact_recipe"]) or {}
        if lam_exact.get("args") and lam_exact.get("env") is not None:   # never the fast images on the exact suite
            exact_classes.append(_lam_class(lam_exact, LAM, LAM["exact_recipe"], True))
    suites = dict(table.get("suites") or {})
    old_suite = suites.get("hrrrv5", {})
    memory = old_suite.get("memory")
    common_suite = {"radiation_s": old_suite.get("radiation_s", 1800), "memory": memory, "kset_repin": None,
                    "window_class_above": above, "window_class_missing_limited_area": LAM_ABOVE, "latest_id": lid}
    suites["hrrrv5"] = {
        "what": "rows made on demand for the hrrrv5 suite on any mesh, fast composition: the template row's runner "
                "arguments and derivative images, the window class of the mesh's size, and the mesh's files, "
                "timestep and memory need",
        "template_row": "hrrrv5-any", "window_classes": fast_classes, **common_suite}
    suites["hrrrv5-exact"] = {
        "what": ("the same on the exact composition (the MYNN-EDMF plume and the mountain-drag closure "
                 "differentiated, the TEMPO adjoint taped; on a global mesh radiation, the transport limiter's "
                 "coefficients and two set-up kernels held, as in the fast composition; on a limited-area mesh the "
                 "transport limiter's coefficients and the lateral-boundary driving data held, radiation live); the "
                 "mountain-drag closure being differentiated, on a global mesh read its gradients as gradients for "
                 "windows of up to about an hour (the template's words)"),
        "template_row": "hrrrv5-any-exact", "window_classes": exact_classes, **common_suite}
    for name in ("hrrrv5", "hrrrv5-exact"):   # measured class memory rules survive a re-point at a newer set
        for new, was in zip(suites[name]["window_classes"], (table.get("suites") or {}).get(name, {}).get("window_classes") or []):
            if (new["memory"] is None and was.get("max_cells") == new["max_cells"]
                    and was.get("mesh_kind") == new.get("mesh_kind")):
                new["memory"] = was.get("memory")
    carry_choices(table, rows, suites, lid)   # a re-run on the same published set keeps the checked words and images
    out = dict(table)
    out["rows"], out["suites"] = rows, suites
    out = add_bounded_lam(out)   # the limited-area class sized to the mesh and the card, after the whole 3 km class
    suites = out["suites"]
    own = {c["assets"]["refuse"]: c["refuse_list"] for s in suites.values() for c in s.get("window_classes") or []
           if c.get("whole") and "refuse" in c["assets"]}   # a full-physics class refuses no kind
    return {"table": out, "refuse": refuse, "class_refuse": own}


def asset_root(root: Path, latest: Path, refuse: str, params: Path | None, class_refuse: dict | None = None) -> None:
    rows = root / "rows" / "hrrrv5-any"
    rows.mkdir(parents=True, exist_ok=True)
    link = root / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(latest.resolve(), link)   # the id's own folder, so a later LATEST does not move a made asset root
    (rows / "refuse.txt").write_text(refuse + "\n", encoding="utf-8")
    for rel, names in (class_refuse or {}).items():   # a whole class's own list (empty: nothing refused)
        (root / rel).write_text(names + "\n", encoding="utf-8")
    if params is not None:
        (rows / "params.json").write_text(params.read_text(encoding="utf-8"), encoding="utf-8")
    (root / "ASSET-ROOT.json").write_text(json.dumps({"latest": str(latest.resolve())}, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--table", type=Path, help="ad-doors.v1.json to rewrite in place")
    ap.add_argument("--asset-root", type=Path, help="asset root to make (needs --latest)")
    ap.add_argument("--latest", type=Path, help="the published tree the asset root points at")
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("--bounded-lam-only", action="store_true",
                    help="with --table and no --manifest: add (or remake) only the bounded limited-area class, every "
                         "other byte of the table as it is")
    a = ap.parse_args()
    if a.bounded_lam_only:
        if a.table is None or a.manifest is not None:
            ap.error("--bounded-lam-only takes --table and no --manifest")
        made = add_bounded_lam(json.loads(a.table.read_text(encoding="utf-8")))
        a.table.write_text(json.dumps(made, indent=2) + "\n", encoding="utf-8")
        print(f"{a.table}: bounded limited-area class {LAM_BOUNDED['suffix']} after each whole limited-area class")
        return 0
    if a.manifest is None:
        ap.error("--manifest is needed")
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    if a.table:
        table = json.loads(a.table.read_text(encoding="utf-8"))
        made = build(manifest, table)
        a.table.write_text(json.dumps(made["table"], indent=2) + "\n", encoding="utf-8")
        print(f"{a.table}: rows hrrrv5-any, hrrrv5-any-exact and their window classes follow {manifest['id']}")
    if a.asset_root:
        if a.latest is None:
            ap.error("--asset-root needs --latest")
        refuse = _split(manifest["recipes"][CLASSES[0]["recipe"]])["refuse"]   # build() checks every recipe shares it
        table = json.loads(a.table.read_text(encoding="utf-8")) if a.table else None
        own = build(manifest, table)["class_refuse"] if table is not None else {}
        asset_root(a.asset_root, a.latest, refuse, a.params, own)
        print(f"{a.asset_root}: latest -> {a.latest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
