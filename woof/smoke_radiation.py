"""Explicit solar and thermal feedback from transported aerosol optics."""
from __future__ import annotations


def apply_smoke_radiation(document: dict, mode: str) -> dict:
    """Set the radiation choice in an experiment document, in place.

    ``on`` selects live aerosol optics in place of climatology and the separate
    shortwave-only addition. It leaves transport, sources, cloud activation and
    the chosen radiation solvers alone. ``off`` disables both live feedback
    paths while retaining the selected background aerosol parameterization.
    """
    from types import SimpleNamespace
    from woof.chem_table import load
    from woof.config import ConfigurationRefused

    if mode not in ("on", "off"):
        raise ValueError("smoke radiation must be 'on' or 'off'")
    shared = document.get("shared", {})
    # Validate every effective domain before changing any part of the document.
    if mode == "on":
        for domain in document.get("domain", []) or [{}]:
            effective = {**shared, **domain}
            table = load(SimpleNamespace(**effective))
            if table is None or not table.rows_for("optics.gocart"):
                raise ConfigurationRefused(
                    "smoke-radiation-no-optics: --smoke-radiation on needs "
                    "active transported smoke/aerosol rows with optics.gocart; "
                    "otherwise no aerosol can supply the requested radiation "
                    "feedback. Select chem_sets with optical mass rows and "
                    "their initial, boundary or emission sources first.")
    settings = {"aer_ra_feedback": int(mode == "on"), "simple_dir_fdb": 0}
    if mode == "on":
        settings.update(aer_opt=0, aer_op_opt=1)
    document.setdefault("shared", {}).update(settings)
    # These chemistry settings have shared scope in the experiment schema.
    # Normal validation still refuses invalid per-domain chemistry overrides.
    return document


def at_flag(args, config, payload):
    """Restate the opt-in in a per-run config before admission/preparation."""
    mode = getattr(args, "smoke_radiation", None)
    if mode is None:
        return config, payload
    from copy import deepcopy
    from woof.go_cli import GoRefusal, _prepared_launch
    if _prepared_launch(args):
        raise GoRefusal(
            "smoke-radiation-prepared-identity: --smoke-radiation changes the "
            "physics configuration bound into prepared inputs or a restart; "
            "ignoring it would run different feedback than requested. Apply "
            "it when preparing: woof go CONFIG --smoke-radiation " + mode)
    from woof.companion_setups import _resolve_declared_paths
    from woof.case_data import config_fetch_out
    from woof.core.preflight import config_forcing_source
    from woof.smoke_execution import _restate
    source = config_forcing_source(config, priced_only=False)
    raw = _resolve_declared_paths(deepcopy(payload), base_dir=config.parent, source=source)
    if raw.get("fetch"):
        raw["fetch"]["out"] = str(config_fetch_out(payload, base_dir=config.parent))
    apply_smoke_radiation(raw, mode)
    out, loaded = _restate(args, config, payload, raw,
                           "smoke-radiation-" + mode, source=source)
    detail = ("live SW/LW aerosol optics; aer_opt=0, aer_op_opt=1, simple_dir_fdb=0"
              if mode == "on" else "live aerosol feedback disabled; transport retained")
    print(f"go: --smoke-radiation {mode}: {detail} ({out})", flush=True)
    return out, loaded
