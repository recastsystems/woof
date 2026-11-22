"""The ArWen-route NoahMP census check across engine generations.

THE BREAKAGE THIS GUARDS (hex 0.5.0 on the gpuwm 3.0.0 candidate 15370d23f9, 2026-11-10):
gpuwm 3.0 publishes the census counts as 0-d arrays and labels its device-resident glacier path
``noahmp-glacier/resident-cuda``.  hex compared the label to the 2.9 kernel-file label and then
JSON-dumped the counts into its step receipt, so every ArWen-route forecast stopped at step 0.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from woof.hex import cuda_arwen_physics_v841 as m

EXPECTED = {
    "land": 10155,
    "water": 27330,
    "sea_ice": 2140,
    "glacier": 1337,
    "glacier_path": m._GLACIER_CUDA_PROVENANCE,
}


def _validate(monkeypatch: pytest.MonkeyPatch, published: dict, expected: dict | None = None) -> dict:
    layout = dict(EXPECTED if expected is None else expected)
    monkeypatch.setattr(
        m.PersistentTwoPhaseCudaPhysicsBackendV841,
        "noahmp_routing_census",
        lambda self: dict(layout),
    )
    fake = types.SimpleNamespace(_seam=types.SimpleNamespace(last_noahmp_census=published))
    return dict(m.PersistentTwoPhaseCudaPhysicsBackendV841._validate_noahmp_census(fake, require=True))


def test_the_2_9_census_is_accepted_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _validate(monkeypatch, dict(EXPECTED)) == EXPECTED


def test_the_3_0_census_is_accepted_and_is_plain_json(monkeypatch: pytest.MonkeyPatch) -> None:
    published = {key: (np.array(value) if isinstance(value, int) else value) for key, value in EXPECTED.items()}
    published["glacier_path"] = "noahmp-glacier/resident-cuda"
    got = _validate(monkeypatch, published)
    # Published under the canonical name, which the step receipt check compares against.
    assert got["glacier_path"] == m._GLACIER_CUDA_PROVENANCE
    assert all(type(got[key]) is int for key in ("land", "water", "sea_ice", "glacier"))
    # The whole published census equals the 2.9 one, so the downstream check of the step receipt
    # (require_arwen_v2_surface_execution) sees the census it expects at every step.
    assert got == EXPECTED
    json.dumps(got)


@pytest.mark.parametrize("path", ["noahmp-glacier/host (woof/core/noahmp_glacier.py)", "noahmp-glacier/bound:x"])
def test_a_host_or_unknown_glacier_path_is_still_refused(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    with pytest.raises(ValueError, match="census/provenance"):
        _validate(monkeypatch, {**EXPECTED, "glacier_path": path})


def test_a_count_that_differs_is_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="census/provenance"):
        _validate(monkeypatch, {**EXPECTED, "glacier": np.array(1336), "glacier_path": "noahmp-glacier/resident-cuda"})


def test_the_3_0_census_of_a_region_without_glacier_drops_the_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # gpuwm 3.0 names its glacier path on a region with no glacier column; the published census, like 2.9's, does not.
    # THE BREAKAGE: the both-routes matrix ArWen region (290 land, 0 glacier) stopped at step 0 (2026-11-10).
    no_glacier = {"land": 290, "water": 0, "sea_ice": 0, "glacier": 0}
    published = {**{key: np.array(value) for key, value in no_glacier.items()},
                 "glacier_path": "noahmp-glacier/resident-cuda"}
    assert _validate(monkeypatch, published, expected=no_glacier) == no_glacier


def test_the_2_9_census_of_a_region_without_glacier_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    no_glacier = {"land": 290, "water": 0, "sea_ice": 0, "glacier": 0}
    assert _validate(monkeypatch, dict(no_glacier), expected=no_glacier) == no_glacier
