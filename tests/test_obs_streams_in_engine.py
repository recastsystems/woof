"""The observation modules the engine's own data assimilation uses live in the engine.

THE BREAKAGE THIS PREVENTS.  Through 2.9.0's first rounds, ``woof.da.obs_point``,
``gpuwm.local_da_fetch`` and ``gpuwm.local_da_observations`` imported
``woof.globe`` (woof global) for the neutral observation table, two column
operators, the stream registry and five front-door rows.  Without woof global a
neutral table was refused, every registered stream was reported unavailable, and
the tests that covered them skipped on every installation; with woof global
superseded by hex global, that dependency would have pointed at a package being
retired.  These tests keep it from coming back and pin the moved door rows to
the binaries' own contracts.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "woof"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def test_no_engine_module_imports_the_global_package():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if any(name == "arwen_global" or name.startswith("arwen_global.") for name in _imports(path)):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"engine modules import woof global: {offenders}"


def test_every_stream_door_is_an_engine_row():
    from woof.obs.frontdoor import door_by_binary
    from woof.obs.streams import STREAMS

    for spec in STREAMS.values():
        if spec.door is not None:
            assert door_by_binary(spec.door).name == spec.door


def test_an_unknown_door_is_refused_by_name():
    from woof.obs.frontdoor import door_by_binary

    with pytest.raises(KeyError, match="rw_igra2"):
        door_by_binary("rw_nonesuch")


CRATE_BINS = ROOT / "tools" / "rustwx" / "crates" / "rw-obs" / "src" / "bin"


@pytest.mark.parametrize("key,source", [
    ("igra2", "igra2.rs"), ("amv", "amv.rs"), ("ndbc", "ndbc.rs"),
    ("gnssro", "gnssro.rs"), ("wis2", "wis2.rs"), ("madis", "madis.rs")])
def test_stream_door_markers_are_the_binaries_own_abi_lines(key, source):
    from woof.obs.frontdoor import STREAM_DOORS

    path = CRATE_BINS / source
    if not path.is_file():
        pytest.skip(f"{path} is not in this checkout (a wheel carries no crate sources)")
    text = path.read_text(encoding="utf-8")
    match = re.search(r'const ABI_MARKER: &str = "((?:[^"\\]|\\.)*)";', text, re.S)
    assert match, f"no ABI_MARKER in {source}"
    literal = re.sub(r"\\\n\s*", "", match.group(1)).replace("\\t", "\t")
    assert STREAM_DOORS[key].abi_marker == literal


def test_the_neutral_table_round_trips_through_the_engine_decoder(tmp_path):
    import csv

    from woof.obs import neutral_table

    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(neutral_table.TABLE_HEADER)
        writer.writerow(["igra2", "USM00072451", 37.76, -99.97, 790.0, 50000, "2026-10-12T12:03:00Z",
                         "temperature_k", 258.4, 1.2, "sonde_level", "2026-10-12T12:00:00Z", "", "", ""])
    source, rows, counts = neutral_table.decode_neutral_csv(path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert (row.variable, row.value, row.level_pa) == ("temperature_k", 258.4, 50000.0)


def test_the_column_operators_are_the_moved_formulas():
    import numpy as np

    from woof.obs.column_operators import dewpoint_from_specific_humidity, refractivity_n

    # dry air: N = 77.6 p/T
    assert float(refractivity_n(100000.0, 300.0, 0.0)) == pytest.approx(77.6 * 1000.0 / 300.0)
    # saturated at 20 C and 1000 hPa (q about 0.01468) has a dewpoint near 20 C
    assert float(dewpoint_from_specific_humidity(0.01468, 100000.0)) == pytest.approx(293.15, abs=0.2)
    assert np.isfinite(dewpoint_from_specific_humidity(np.zeros(3), 100000.0)).all()


def test_gpuwm_obs_streams_is_a_command():
    import argparse

    from woof.obs.cli import register_cli

    parser = argparse.ArgumentParser(prog="woof")
    register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["obs", "streams", "streams"])
    assert callable(args.func)
