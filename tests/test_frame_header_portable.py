"""The frame header carries a machine-independent identity.

The breakage this names: a canonical frame header quotes, per field, the
sha256 of that field's array bytes AND the absolute paths the arrays came
from.  Two boxes decoding the same published bytes therefore produce two
different header digests, for two reasons that are both facts about the
box and not about the decode:

1. the input paths differ (a checkout, a staging tree, a drive letter);
2. a field produced through a TRANSCENDENTAL takes its last bits from
   the box's libm.  Measured 2026-08-20, same bytes on the Windows
   desktop (UCRT) and test host 1 (glibc): on the netCDF sample
   committed beside the crate goldens, 35 of 38 arrays are identical to
   the bit and the three the two ``exp``-based humidity derivations
   produce are at most 3 ULP apart -- 4.1e-16 relative; on the Lambert
   case, all four wind components move, because the grid-relative wind
   rotation is ``sin``/``cos``.

So a golden recorded on one box fails on another while every decoded
number is right.  The remedy is a second, DECLARED digest that both
engines emit beside the raw one: input paths reduced to basenames, and
each libm-dependent field's array digest replaced by ``libm:<name>``.
The raw digest keeps its meaning -- what THIS box measured -- and the
portable one is what a golden may assert.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from woof import mapped_engine_bridge as engine_bridge
from woof.source_frame import (
    PORTABLE_HEADER_RULE,
    portable_frame_header,
    portable_frame_header_sha256,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SAMPLE = (
    REPOSITORY
    / "tools" / "rw_wps" / "crates" / "mapped-engine" / "tests" / "goldens"
    / "netcdf-pressure-level-sample.nc"
)
MAPPING = REPOSITORY / "configs" / "rw-wps-era5-netcdf.mapping.json"


def _header(sample: str, humidity_digest: str) -> dict[str, object]:
    """A header shaped like the real one, with one direct and one derived field.

    ``sample`` is the input file spelled exactly as the engine was handed
    it, because that is the spelling the mask has to match.
    """

    return {
        "schema": "gpuwm.source_frame.v1",
        "source_id": "unit",
        "source_cycle": "2026-08-20T00:00:00+00:00",
        "grid": {"projection": "regular_latitude_longitude", "nx": 9, "ny": 6},
        "vertical_coordinates": {},
        "initialization_policies": {},
        "fields": [
            {
                "canonical_name": "air_temperature",
                "data_reference": "sha256:aaaa",
                "source_field": f"{sample}:t",
                "dtype": "<f8",
                "shape": [6, 9],
            },
            {
                "canonical_name": "specific_humidity",
                "data_reference": f"sha256:{humidity_digest}",
                "source_field": f"{sample}:r;{sample}:t",
                "dtype": "<f8",
                "shape": [6, 9],
            },
        ],
    }


def test_the_portable_digest_ignores_where_the_inputs_live():
    left = _header("/srv/inputs/staging/sample.nc", "bbbb")
    right = _header(r"D:\inputs\staging\sample.nc", "bbbb")
    assert portable_frame_header_sha256(
        left, inputs=["/srv/inputs/staging/sample.nc"], libm_dependent={"specific_humidity"}
    ) == portable_frame_header_sha256(
        right, inputs=[r"D:\inputs\staging\sample.nc"], libm_dependent={"specific_humidity"}
    )


def test_the_portable_digest_ignores_a_derived_field_s_last_bits():
    """A derived array's digest is a libm identity, so it is elided by name."""

    inputs = ["/staging/sample.nc"]
    left = _header("/staging/sample.nc", "bbbb")
    right = _header("/staging/sample.nc", "cccc")
    assert portable_frame_header_sha256(
        left, inputs=inputs, libm_dependent={"specific_humidity"}
    ) == portable_frame_header_sha256(
        right, inputs=inputs, libm_dependent={"specific_humidity"}
    )
    # And the elision is by NAME, so a derived field that vanished or was
    # renamed still moves the digest.
    renamed = copy.deepcopy(left)
    renamed["fields"][1]["canonical_name"] = "specific_humidity_2m"
    assert portable_frame_header_sha256(
        renamed, inputs=inputs, libm_dependent={"specific_humidity_2m"}
    ) != portable_frame_header_sha256(
        left, inputs=inputs, libm_dependent={"specific_humidity"}
    )


def test_a_direct_field_s_bytes_still_move_the_portable_digest():
    """The gate this keeps: a decoded array that changed is still caught."""

    inputs = ["/staging/sample.nc"]
    left = _header("/staging/sample.nc", "bbbb")
    right = copy.deepcopy(left)
    right["fields"][0]["data_reference"] = "sha256:zzzz"
    assert portable_frame_header_sha256(
        left, inputs=inputs, libm_dependent={"specific_humidity"}
    ) != portable_frame_header_sha256(
        right, inputs=inputs, libm_dependent={"specific_humidity"}
    )


def test_the_portable_header_names_its_rule():
    portable = portable_frame_header(
        _header("/staging/sample.nc", "bbbb"),
        inputs=["/staging/sample.nc"],
        libm_dependent={"specific_humidity"},
    )
    assert portable["portable_rule"] == PORTABLE_HEADER_RULE
    assert portable["fields"][1]["data_reference"] == "libm:specific_humidity"
    assert portable["fields"][0]["data_reference"] == "sha256:aaaa"
    assert portable["fields"][0]["source_field"] == "sample.nc:t"


def _materialization(engine: str, sample: Path) -> dict[str, object]:
    environment = dict(os.environ)
    environment[engine_bridge.ENGINE_ENV] = engine
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import json,sys;from woof.mapped_source import inspect_mapped_source;"
            "print(json.dumps(inspect_mapped_source(sys.argv[1], [sys.argv[2]])"
            "['materialization']))",
            str(MAPPING), str(sample),
        ],
        capture_output=True, text=True, env=environment, cwd=str(REPOSITORY),
    )
    if completed.returncode:
        pytest.fail(
            f"{engine} engine refused on the committed sample: "
            f"{completed.stderr.strip().splitlines()[-1:] or ['(no stderr)']}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_two_engines_agree_on_the_portable_digest():
    """The rule is one rule: a port that drifts is a silent parity hole."""

    if not SAMPLE.is_file():  # pragma: no cover - the sample is committed
        pytest.skip(f"the committed netCDF sample is absent: {SAMPLE}")
    rust = _materialization(engine_bridge.ENGINE_RUST, SAMPLE)
    python = _materialization(engine_bridge.ENGINE_PYTHON, SAMPLE)
    assert rust["portable_rule"] == python["portable_rule"] == PORTABLE_HEADER_RULE
    assert (
        rust["frame_header_sha256_portable"]
        == python["frame_header_sha256_portable"]
    )


@pytest.mark.parametrize("engine", [engine_bridge.ENGINE_RUST, engine_bridge.ENGINE_PYTHON])
def test_both_engines_publish_a_portable_digest_that_survives_a_move(engine, tmp_path):
    """The artifact proof: the same bytes at two paths, one portable digest.

    This is the golden's failure in miniature -- the raw digest changes
    because the file moved, and nothing about the decode did.
    """

    if not SAMPLE.is_file():  # pragma: no cover - the sample is committed
        pytest.skip(f"the committed netCDF sample is absent: {SAMPLE}")

    digests: list[tuple[list[str], list[str]]] = []
    for name in ("left", "right"):
        directory = tmp_path / name
        directory.mkdir()
        copied = directory / SAMPLE.name
        shutil.copy2(SAMPLE, copied)
        materialization = _materialization(engine, copied)
        assert materialization["verdict"] == "PASS"
        digests.append(
            (
                materialization["frame_header_sha256"],
                materialization["frame_header_sha256_portable"],
            )
        )

    (left_raw, left_portable), (right_raw, right_portable) = digests
    assert left_raw != right_raw, (
        "the raw digest is supposed to be an identity of THIS run's paths; "
        "if it no longer moves with them this test is measuring nothing"
    )
    assert left_portable == right_portable
