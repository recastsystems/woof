"""The go-launch step of TC initialization: restate a launch on its canonical source row.

Kept out of :mod:`woof.storm_merge` because it re-times the saved setup through
woof.companion_setups, a forecast-launch module the standalone preparation
package does not ship; storm_merge itself is part of preparation.
"""
from __future__ import annotations

from pathlib import Path

from woof.storm_merge import preparation_source


def prepare_launch(args, config, payload):
    """Restate an explicitly transformed launch using its table-selected row."""
    if not getattr(args, "storm_merge", None) and getattr(args, "vortex_init", "off") == "off":
        return config, payload
    if getattr(args, "prepared_root", None) or getattr(args, "restart", None):
        raise ValueError("TC initialization must run during preparation, before a prepared forecast or restart")
    fetch = payload.get("fetch", {})
    source = fetch.get("source")
    if source is None:
        raise ValueError("TC initialization requires a declared fetch source")
    selected = preparation_source(source)
    if source == selected:
        return config, payload
    import tomllib
    from woof.companion_setups import retime_to_cycle
    from woof.fetch import parse_cycle, fetch_accepts_area
    from woof.toml_document import emit_experiment_toml
    root = Path(args.outdir) if getattr(args, "outdir", None) else Path(config).parent / f"{Path(config).stem}-go"
    args.outdir = root
    output = root / "canonical-source" / Path(config).name
    retime_to_cycle(config, parse_cycle(str(fetch["cycle"]), source), output)
    raw = tomllib.loads(output.read_text())
    raw["fetch"]["source"] = selected
    if not fetch_accepts_area(selected):
        raw["fetch"].pop("area", None)
    # Acquisition directories are source-specific. Reusing a legacy subset
    # manifest here could silently omit the hydrometeor inventory.
    raw["fetch"].pop("source_root", None)
    raw["fetch"]["out"] = str((root / "canonical-forcing").resolve())
    if getattr(args, "data_dir", None):
        raise ValueError("storm merge's canonical source needs its own full-field fetch directory; omit --data-dir or select the canonical source row explicitly")
    text = emit_experiment_toml(raw)
    output.write_text(text)
    args.config = output
    print(f"go: TC initialization selects {selected} from {source}'s canonical preparation row: {output}")
    return output, raw
