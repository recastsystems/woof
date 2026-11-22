# SPDX-License-Identifier: Apache-2.0
"""Learned and symbolic surface and boundary-layer closures on the RRFS hrrrv5 route (RUC land + MYNN-EDMF). OFF by default.

Three options, each its own switch, at most one per run:

==========================  ==========================================  ================================================
option                      door (``--physics KEY=VALUE``)              word
==========================  ==========================================  ================================================
surface-flux net            ``config_surface_flux_net=r4_a50``          ``GPUWM_HEX_SURFACE_FLUX_NET=r4_a50``
neural boundary layer       ``config_neural_pbl=v1`` (or ``v2``)        ``GPUWM_HEX_NEURAL_PBL=v1`` (or ``v2``)
  its dry-air gate          ``config_neural_pbl_dry_gate=true`` (v1->v2) ``GPUWM_HEX_NEURAL_PBL_DRY_GATE=1``
8-parameter closure         ``config_autophysics_pbl=v2``,              ``GPUWM_HEX_AUTOPHYSICS_PBL=v2``, ``v3_bowen``,
                            ``v3_bowen`` or ``v2_warm_night``           ``v2_warm_night``
==========================  ==========================================  ================================================

``on`` names an option's first set (r4_a50, v1, v2); ``off`` (or unset) is the plain model.  Off, nothing here imports
torch or cupy, no file is read and the two slots run exactly the code they ran before the options existed: each slot
reads one module attribute (``ACTIVE``) per call and finds None.

Where each acts (the slots' own code):
  * after RUC's land entry (``land_ruc_land``), before the surface diagnostics and MYNN-EDMF: the surface-flux net,
    the neural boundary layer's surface subnet and the 8-parameter closure shift sensible and latent heat flux (and
    the moisture flux with it) and, for the diagnostics and the boundary layer only, the skin temperature and surface
    humidity; the skin shift is undone before the next land step, so RUC's soil heat balance never carries it.
  * after MYNN-EDMF: the neural boundary layer's nocturnal subnet adds a column-conserving vertical redistribution of
    theta, qv, u and v over the lowest ten levels (no column sum changes).

Weights and parameter sets are package data (``woof/hex/data/closures``) named with their SHA-256 in
``closures.v1.json``; a file whose sum differs is refused at the start, before any step.  The two nets run in PyTorch
on the card (torch is imported at the first step, after the engine has loaded its kernel pack, so the two do not
load different NVRTC libraries); the 8-parameter closure is cupy only and never imports torch.

The networks, their checkpoints, the eight parameters and the gates are the project's own (November 2026).  NOTICE
section 20 names each option's training data.  ``v2_warm_night`` is the v2 numbers with the cold-night branch of
the regime gate removed (the v2 nocturnal cooling only on warm nights, T0 above 20 C).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from dataclasses import replace
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

TABLE = "closures.v1.json"

KEY_SFN = "config_surface_flux_net"
KEY_NPBL = "config_neural_pbl"
KEY_NPBL_GATE = "config_neural_pbl_dry_gate"
KEY_AP = "config_autophysics_pbl"
KEYS = (KEY_SFN, KEY_NPBL, KEY_NPBL_GATE, KEY_AP)

WORD_SFN = "GPUWM_HEX_SURFACE_FLUX_NET"
WORD_NPBL = "GPUWM_HEX_NEURAL_PBL"
WORD_NPBL_GATE = "GPUWM_HEX_NEURAL_PBL_DRY_GATE"
WORD_AP = "GPUWM_HEX_AUTOPHYSICS_PBL"
WORDS = {KEY_SFN: WORD_SFN, KEY_NPBL: WORD_NPBL, KEY_NPBL_GATE: WORD_NPBL_GATE, KEY_AP: WORD_AP}

OPTION_OF = {KEY_SFN: "surface_flux_net", KEY_NPBL: "neural_pbl", KEY_AP: "autophysics_pbl"}

#: The one closure installed in this process (install), read by the RUC land slot and the MYNN-EDMF slot each call.
#: None is the plain model.
ACTIVE: Any = None


class ClosureRefusal(ValueError):
    """A closure selection the run cannot honour, refused by name before any step."""


def _data_dir() -> Path:
    return Path(str(resources.files("woof.hex") / "data" / "closures"))


def load_table() -> dict[str, Any]:
    return json.loads((_data_dir() / TABLE).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(name: str, expected: str, directory: Path | None = None) -> Path:
    """The package-data file ``name`` after its SHA-256 is checked against the table's, or a refusal naming both."""
    path = (directory or _data_dir()) / name
    if not path.is_file():
        raise ClosureRefusal(f"closure file {name} is not in {path.parent}: the wheel or tree is incomplete")
    got = _sha256(path)
    if got != expected.lower():
        # THE BREAKAGE THIS PREVENTS: a changed or truncated checkpoint or parameter set would run under the option's
        # name and its recorded scores.
        raise ClosureRefusal(f"closure file {path} has SHA-256 {got}, the table names {expected.lower()}: refused")
    return path


def _text(value: Any) -> str:
    return str(value).strip().strip("\"'").strip().casefold()


def _logical(key: str, value: Any) -> bool:
    text = _text(value).strip(".")
    if text in ("true", "t", "1", "on", "yes"):
        return True
    if text in ("false", "f", "0", "off", "no", ""):
        return False
    raise ClosureRefusal(f"{key} must be true or false, not {value!r}")


def _set_name(table: Mapping[str, Any], option: str, key: str, value: Any) -> str | None:
    row = table["options"][option]
    text = _text(value)
    if text in ("", "off", "none", "false", "0"):
        return None
    if text in ("on", "true", "1"):
        return row["first_set"]
    text = row.get("aliases", {}).get(text, text)
    if text not in row["sets"]:
        raise ClosureRefusal(f"{key}={value!r}: not off or one of {sorted(row['sets'])} ({WORDS[key]} takes the same)")
    return text


def selection(explicit: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None) -> dict | None:
    """The run's closure from the door's ``--physics`` keys and the GPUWM_HEX_* words, or None (the plain model).

    Refusals, each by name: an unknown key or value; a door key and a word that name different sets; the dry-air
    gate without the neural boundary layer, or against an explicit v1; two options on at once."""
    environ = os.environ if environ is None else environ
    explicit = {k.strip().casefold(): v for k, v in dict(explicit or {}).items()}
    unknown = sorted(set(explicit) - set(KEYS))
    if unknown:
        raise ClosureRefusal(f"unknown closure options {unknown}; known: {list(KEYS)}")
    named: dict[str, tuple[Any, str]] = {}
    for key in KEYS:
        word = environ.get(WORDS[key])
        if key in explicit and word not in (None, ""):
            if _text(explicit[key]) != _text(word):
                raise ClosureRefusal(f"--physics {key}={explicit[key]} and {WORDS[key]}={word} name two selections "
                                     "for one run; give one")
        if key in explicit:
            named[key] = (explicit[key], "door")
        elif word not in (None, ""):
            named[key] = (word, "word")
    if not named:
        return None
    table = load_table()
    chosen: dict[str, tuple[str, str]] = {}
    for key in (KEY_SFN, KEY_NPBL, KEY_AP):
        if key in named:
            name = _set_name(table, OPTION_OF[key], key, named[key][0])
            if name is not None:
                chosen[OPTION_OF[key]] = (name, named[key][1])
    if KEY_NPBL_GATE in named:
        gate = _logical(KEY_NPBL_GATE, named[KEY_NPBL_GATE][0])
        if "neural_pbl" not in chosen:
            if gate:
                # THE BREAKAGE THIS PREVENTS: the gate would be recorded and never run.
                raise ClosureRefusal(f"{KEY_NPBL_GATE} ({WORD_NPBL_GATE}) gates the neural boundary layer; this run "
                                     f"has none ({KEY_NPBL} / {WORD_NPBL} is off)")
        else:
            current, source = chosen["neural_pbl"]
            gated = table["options"]["neural_pbl"]["dry_gate_set"]
            ungated = table["options"]["neural_pbl"]["first_set"]
            if gate and current == ungated:
                chosen["neural_pbl"] = (gated, source)
            elif gate != (current == gated):
                raise ClosureRefusal(f"{KEY_NPBL}={current} and {KEY_NPBL_GATE}={'true' if gate else 'false'} "
                                     f"disagree: {gated} is {ungated} with the dry-air gate on")
    if len(chosen) > 1:
        names = " and ".join(f"{option} {name}" for option, (name, _) in sorted(chosen.items()))
        # THE BREAKAGE THIS PREVENTS: each option shifts RUC's land fluxes and saves the skin temperature and
        # surface humidity to restore them before the next land step.  Two at once correct the same fluxes twice
        # (each was trained against plain RUC), and the second saves the first's shifted skin, so the restore leaves
        # a shift in the soil heat balance that neither carries.
        raise ClosureRefusal(f"{names} cannot run together: both shift RUC's land fluxes and skin temperature on "
                             "the same slot. Turn one off")
    if not chosen:
        return None
    (option, (name, source)), = chosen.items()
    row = table["options"][option]
    choice = dict(row["sets"][name])
    files = {}
    for kind in ("weights", "parameters"):
        if kind in choice:
            files[choice[kind]] = table["files"][choice[kind]]
    return {"option": option, "set": name, "source": source, "torch": bool(row["torch"]),
            "slots": list(row["slots"]), **choice, "files": files}


def configure(config: Any, explicit: Mapping[str, Any] | None = None,
              environ: Mapping[str, str] | None = None) -> Any:
    """The route's DriverConfig with the run's closure on it (``config.closure``), or the config unchanged when
    every option is off.  Refused by name: a closure on a route without RUC land (and MYNN-EDMF, for the neural
    boundary layer), or with no MPAS suite route at all."""
    choice = selection(explicit, environ)
    if choice is None:
        return config
    label = f"{choice['option']} {choice['set']}"
    if config is None:
        raise ClosureRefusal(f"the closure {label} runs on the MPAS suite route's RUC land slot (--physics-suite "
                             "hrrrv5); this run has no MPAS suite route")
    table_key = str(getattr(config, "table_key", "mpasphys_v842"))
    if "rrfs" not in table_key:
        # THE BREAKAGE THIS PREVENTS: the MPAS-A v8.4.2 route runs its land surface and boundary layer in other slots,
        # which carry no hook, so the option would be recorded and never run.
        raise ClosureRefusal(f"the closure {label} runs on the RRFS route's resident RUC and MYNN-EDMF slots "
                             f"(--physics-suite hrrrv5); this run's route is {table_key}")
    selectors = getattr(config, "selectors", {})
    active = getattr(config, "active", lambda slot: selectors.get(slot, "off") != "off")
    if selectors.get("land_surface") != "sf_ruc" or not active("land_surface"):
        # THE BREAKAGE THIS PREVENTS: the option would be recorded and never run (its hook is RUC's land entry).
        raise ClosureRefusal(f"the closure {label} acts after RUC's land step; this run's land surface is "
                             f"{selectors.get('land_surface')!r}, not 'sf_ruc'")
    if "pbl" in choice["slots"] and (selectors.get("pbl") != "bl_mynnedmf" or not active("pbl")):
        raise ClosureRefusal(f"the closure {label} acts after MYNN-EDMF; this run's boundary layer is "
                             f"{selectors.get('pbl')!r}, not 'bl_mynnedmf'")
    for slot in choice["slots"]:
        slot_name = "land_surface" if slot == "land" else slot
        backend = str(getattr(config, "backends", {}).get(slot_name, "hex-cuda"))
        if backend != "hex-cuda":
            raise ClosureRefusal(f"the closure {label} runs on the GPU ports' resident slots; config_"
                                 f"{slot_name}_backend is {backend}")
    if choice["torch"] and find_spec("torch") is None:
        # Not imported here: torch first changes the NVRTC the engine compiles with (the first step imports it).
        raise ClosureRefusal(f"the closure {label} runs its network in PyTorch on the card and this environment has "
                             "no torch: pip install torch (CUDA build matching the card's driver), or turn it off")
    return replace(config, closure=MappingProxyType(choice))


def install(config: Any, out: Path | None = None) -> Any:
    """Make ``config.closure`` this process's ACTIVE closure (None: the plain model) and check its files' sums.

    Called once by the forecast route before the first step.  ``out`` is the run's folder: a closure that ran writes
    ``closure.json`` there at exit (calls, evaluations, host milliseconds on the step path)."""
    global ACTIVE
    choice = getattr(config, "closure", None)
    if not choice:
        ACTIVE = None
        return None
    table = load_table()
    paths = {name: check_file(name, sha) for name, sha in choice["files"].items()}
    option = choice["option"]
    if option == "surface_flux_net":
        from .surface_flux_net import SurfaceFluxNetHook
        hook = SurfaceFluxNetHook(checkpoint=paths[choice["weights"]], scale=choice["scale"],
                                  stride=choice["stride"], tsk_couple=choice["tsk_couple"],
                                  tsk_eq_weight=choice["tsk_eq_weight"])
    elif option == "neural_pbl":
        from .neural_pbl import NeuralPblHook
        hook = NeuralPblHook(checkpoint=paths[choice["weights"]], stride=choice["stride"],
                             dry_gate=choice["dry_gate"], tsk_couple=choice["tsk_couple"],
                             tsk_eq_weight=choice["tsk_eq_weight"])
    elif option == "autophysics_pbl":
        from .autophysics_pbl import AutoPhysicsPblHook, load_parameters
        params = load_parameters(paths[choice["parameters"]], choice["parameter_set"])
        hook = AutoPhysicsPblHook(params=params["params"], night_gate=params["night_gate"], gain=choice["gain"],
                                  stride=choice["stride"], tsk_couple=choice["tsk_couple"],
                                  tsk_eq_weight=choice["tsk_eq_weight"])
    else:  # pragma: no cover - the table names the options
        raise ClosureRefusal(f"closure option {option!r} is not in {TABLE} {sorted(table['options'])}")
    hook.label = f"{option} {choice['set']}"
    print(f"CLOSURE: {hook.label} on ({choice['source']}): {', '.join(f'{n} sha256 {s[:12]}' for n, s in choice['files'].items())}",
          flush=True)
    if out is not None:
        record = Path(out) / "closure.json"

        def _flush(hook=hook, record=record, choice=dict(choice)):
            try:
                record.write_text(json.dumps({"schema": "hexcore.closure-run.v1", "closure": choice,
                                              **hook.counters()}, indent=1, default=str) + "\n")
            except Exception:
                pass
        atexit.register(_flush)
    ACTIVE = hook
    return hook


def needs(slot: str) -> bool:
    """True when the installed closure acts on ``slot`` ('land' or 'pbl')."""
    return ACTIVE is not None and slot in ACTIVE.slots


def require_resident(slot: str, provider: Any) -> None:
    """Called where a RUC land or MYNN-EDMF call found no resident slot (the native library route).

    THE BREAKAGE THIS PREVENTS: the closure's hook is the resident slot's own; on the native library route the run
    would carry the option's name and run the plain model."""
    if needs(slot):
        raise ClosureRefusal(f"the closure {ACTIVE.label} runs on the resident RRFS slots (the ported framework route); "
                             f"this run's {'RUC land' if slot == 'land' else 'MYNN-EDMF'} call runs through the native "
                             "library route")
