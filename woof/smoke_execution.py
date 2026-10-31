"""Visible per-run capacity fallback for the transported smoke package."""
from __future__ import annotations


from woof.smoke_receipts import message, receipt, config_receipt


#: HRRR-Smoke's whole chem key set (HRRR v4 hrrr_wrf.nl &chem, &dynamics), the
#: one place the opt-in restores it from: the packaged HRRR v4 recipe ships
#: smoke off in 2.9.4 (it costs ~11x per step on full CONUS, measured on 8x
#: RTX 5090, fold f1a3e8609b), and ``--smoke`` on ``woof go``, ``woof domain
#: --recipe``, ``woof standby`` and ``woof race`` applies exactly this.
HRRR_SMOKE_SETTINGS = {
    "chem_sets": "smoke",
    "chem_sources": "rave-3km,rap-native-smoke,hrrr-native-smoke",
    "simple_dir_fdb": 1,
    "biomass_burn_opt": 4,
    "enh_vermix": True,
    "flam_part": 0.75,
    "aer_drydep_opt": 111,
    "chem_adv_opt": 2,
    "chem_mix6_off": True,
    "vertmix_onoff": 1,
    "wetscav_onoff": -1,
    "plumerisefire_frq": 60,
    "chem_conv_tr": 0,
}

SMOKE_FLAG_HELP = (
    "turn on HRRR-Smoke (transported smoke, its radiative feedback, RAVE "
    "fires); in 2.9.4 it costs about 11x per step on full CONUS (measured on "
    "8x RTX 5090)")


def apply_smoke_settings(document: dict) -> dict:
    """Turn HRRR-Smoke on in a parsed experiment document (in place)."""
    shared = document.setdefault("shared", {})
    shared.pop("smoke_off_reason", None)
    shared.update(HRRR_SMOKE_SETTINGS)
    for domain in document.get("domain", []):
        domain.pop("smoke_off_reason", None)
        if "chem_sets" in domain:
            domain.update(HRRR_SMOKE_SETTINGS)
    return document


def smoke_declared(payload: dict) -> bool:
    from woof.chem_table import chem_names
    declared = [payload.get("shared", {}).get("chem_sets", "")]
    declared.extend(d.get("chem_sets", "") for d in payload.get("domain", []))
    return any(chem_names(value) == ("smoke",) for value in declared)


def _restate(args, config, payload, raw, subdir, *, source):
    """Write the per-run config (and its route companions) under the outdir."""
    from pathlib import Path
    import tomllib
    from woof.companion_domains import _build, _wps_text
    from woof.hrrr_route_inputs import candidate_companions
    from woof.toml_document import emit_experiment_toml
    if getattr(args, "outdir", None) is None:
        args.outdir = config.parent / f"{config.stem}-go"
    out = Path(args.outdir) / subdir / config.name
    built = _build(raw, out)
    beside = config.with_suffix(".namelist.wps")
    wps = _wps_text(built, beside if beside.is_file() else None,
                    out.with_suffix(".namelist.wps"), raw, len(built.domains),
                    original_domain_ids=[d.grid_id for d in built.domains])
    files = candidate_companions(out, built, wps_text=wps, source=source)
    files.append((out, emit_experiment_toml(raw)))
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    args.config = out
    if getattr(args, "wps_namelist", None) is not None:
        args.wps_namelist = out.with_suffix(".namelist.wps")
    return out, tomllib.loads(out.read_text(encoding="utf-8"))


def at_flag_smoke(args, config, payload):
    """``woof go --smoke``: a per-run config with HRRR_SMOKE_SETTINGS applied.

    The recipe file is never edited.  The capacity fallback (:func:`at_launch`)
    runs after this on the returned config, so a card that cannot hold the
    smoke envelope still drops it visibly.
    """
    from copy import deepcopy
    if not getattr(args, "smoke", False) or smoke_declared(payload):
        return config, payload
    from woof.companion_setups import _resolve_declared_paths
    from woof.case_data import config_fetch_out
    from woof.core.preflight import config_forcing_source
    source = config_forcing_source(config, priced_only=False)
    raw = _resolve_declared_paths(deepcopy(payload), base_dir=config.parent, source=source)
    if raw.get("fetch"):
        raw["fetch"]["out"] = str(config_fetch_out(payload, base_dir=config.parent))
    apply_smoke_settings(raw)
    out, loaded = _restate(args, config, payload, raw, "smoke-on", source=source)
    print(f"go: --smoke: HRRR-Smoke on for this run ({out}); {SMOKE_FLAG_HELP.split('; ', 1)[1]}",
          flush=True)
    return out, loaded


def _resident_bytes(exp, probe, *, forcing_intervals=1, source=None):
    from woof.core.preflight import (
        estimate_experiment, profile_from_device_probe, required_mynn_pricing)
    from woof.boundary_fields import source_boundary_species
    # Match resident admission: workspace width changes speed and can shrink
    # before chemistry is dropped. Never downgrade a shape the gate admits.
    with required_mynn_pricing(exp):
        return estimate_experiment(exp, forcing_intervals=forcing_intervals,
            profile=profile_from_device_probe(probe),
            boundary_species=source_boundary_species(source) if source else ()).peak_envelope_bytes


def capacity_reason(exp, *, probe=None, forcing_intervals=1, source=None):
    """CPU estimate only; an unknown card never earns a capacity decision."""
    from woof.chem_table import chem_names
    if chem_names(exp.root.run.chem_sets) != ("smoke",):
        return None
    if exp.tiles.mode == "on":
        return "tiled streaming cannot own post-step smoke process ledgers"
    if exp.devices.enabled or probe is None:
        return None
    required = _resident_bytes(exp, probe, forcing_intervals=forcing_intervals, source=source)
    free = int(probe["free_bytes"])
    if required > free:
        return (f"resident smoke envelope {required / 2**30:.2f} GiB exceeds "
                f"the card's {free / 2**30:.2f} GiB free")
    return None


def at_launch(args, config, payload):
    """Resolve before fetching; the recipe is never edited by this decision."""
    from copy import deepcopy
    from pathlib import Path
    from dataclasses import replace
    import tomllib
    from woof.chem_table import chem_names
    from woof.experiment import load_experiment
    from woof.core.devices import override_device_count
    from woof.core.preflight import device_memory_probe_subprocess, config_forcing_source
    if not smoke_declared(payload):
        return config, payload
    exp = load_experiment(config)
    if chem_names(exp.root.run.chem_sets) != ("smoke",):
        return config, payload
    exp = replace(exp, devices=override_device_count(exp.devices, getattr(args, "devices", None)))
    probe = (None if exp.devices.enabled or exp.tiles.mode == "on"
             else device_memory_probe_subprocess())
    if probe is not None:
        from woof.core.device_probe import sizing_free_memory
        available, _note = sizing_free_memory(int(probe["free_bytes"]))
        probe = dict(probe, free_bytes=available)
    source = config_forcing_source(config, priced_only=False)
    reason = capacity_reason(exp, probe=probe, source=source)
    if reason is None:
        return config, payload
    if getattr(args, "smoke_radiation", None) == "on":
        from woof.go_cli import GoRefusal
        raise GoRefusal(
            "smoke-radiation-capacity: the requested live aerosol feedback "
            "cannot run after removing its transported smoke. " + reason +
            ". Use a smaller domain or sufficient resident GPU memory.")
    from woof.companion_setups import _resolve_declared_paths
    from woof.companion_domains import _build
    from woof.case_data import config_fetch_out
    raw = _resolve_declared_paths(deepcopy(payload), base_dir=config.parent, source=source)
    if raw.get("fetch"):
        raw["fetch"]["out"] = str(config_fetch_out(payload, base_dir=config.parent))
    shared = raw.setdefault("shared", {})
    # Smoke off is chemistry off: every chem switch the recipe set (the
    # transport/filter controls chem_adv_opt and chem_mix6_off among them)
    # returns to its default, which is what a chemistry-off run and its
    # emitted namelists both say.  Breakage this prevents: the HRRR route
    # refused at prepare on any card that cannot hold the smoke envelope,
    # "the emitted HRRR namelists do not reproduce the emitted config:
    # d01 chem_adv_opt: config 2 vs namelist 1; d01 chem_mix6_off: config
    # True vs namelist False" (a 4090 host, 7b4f252a29, 2026-10-21).
    from woof.config import CHEM_RUN_FIELDS
    for table in [shared, *raw.get("domain", [])]:
        for key in CHEM_RUN_FIELDS:
            if key not in ("chem_sets", "chem_sources", "simple_dir_fdb"):
                table.pop(key, None)
    shared.update(chem_sets="", chem_sources="", simple_dir_fdb=0, smoke_off_reason=reason)
    for domain in raw.get("domain", []):
        if "chem_sets" in domain:
            domain.update(chem_sets="", chem_sources="", simple_dir_fdb=0, smoke_off_reason=reason)
    if getattr(args, "outdir", None) is None:
        args.outdir = config.parent / f"{config.stem}-go"
    out = Path(args.outdir) / "smoke-off" / config.name
    off = _build(raw, out)
    if (probe is not None and exp.tiles.mode == "off"
            and _resident_bytes(off, probe, source=source) > int(probe["free_bytes"])):
        # The requested fallback must actually reach a runnable physics road.
        # Dropping smoke alone still leaves full-CONUS physics above one card.
        raw.setdefault("tiles", {})["mode"] = "auto"
        reason += "; chemistry-off physics also exceeds the card, so automatic tiling is selected"
        shared["smoke_off_reason"] = reason
        for domain in raw.get("domain", []):
            if "smoke_off_reason" in domain:
                domain["smoke_off_reason"] = reason
    out, loaded = _restate(args, config, payload, raw, "smoke-off", source=source)
    print(message(reason), flush=True)
    return out, loaded
