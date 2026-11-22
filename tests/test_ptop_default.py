"""The default model top is 5000 Pa (50 hPa), bounded by source coverage.

The 10000 Pa (100 hPa, ~16 km) default put the damp_opt=3 implicit-damping
layer's base near 10.9 km AGL -- inside anvil and overshoot territory for
deep convection -- so a bare default run smeared exactly the structure a
convection-allowing grid exists to resolve.  Measured on the 2026-08-24
plains A/B (HRRR 18Z, 232x184 at 3 km, arms identical except p_top; RTX
5070 Ti, evaluating commit 8c6a109c3): the 10000 Pa arm's anvil-layer
maximum updrafts plateaued at ~17 m/s from 7-10.5 km and collapsed at the
10.9 km sponge base, while the 5000 Pa arm peaked at 22 m/s with natural
decay near 13 km, carried more >=40 dBZ core cells and ~300-500 m higher
cloud tops, scored equal-or-better MRMS FSS (0.0815 vs 0.0798), and ran
12.9% faster at the same VRAM.  5000 Pa is also WRF v4.6.1's own
p_top_requested Registry default (Registry.EM_COMMON:2275).

The default is bounded per source: an emission must never ask for a model
top its source's certified inventory cannot cover, or a bare `woof
domain --source X` config refuses at preparation with the vertical-
coverage refusal after the user has already paid for the acquisition.
That bound is table data on the source's own registry row
(``certified_source_top_pa``), the same seam as forcing cadence, and it
lifts where the row says the fetch reaches further when asked
(``extendable_source_top_pa``).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from woof import domain_wizard
from woof.source_adapters import (get_source_adapter,
                                   wizard_planable_source_ids)

DEFAULT_PA = 5000.0


def test_wizard_shared_default_is_5000_pa():
    assert domain_wizard.DEFAULT_MODEL_TOP_PA == DEFAULT_PA
    assert domain_wizard._SHARED_GRID_AND_DYNAMICS["p_top"] == DEFAULT_PA


def test_emitted_top_is_5000_for_deep_sources_and_bounded_for_shallow():
    # HRRR's native hybrid column reaches ~20 hPa, so the default stands.
    assert domain_wizard.emitted_model_top_pa("hrrr") == DEFAULT_PA
    # The certified GFS pressure ladder stops at 100 hPa, but every door
    # that downloads for a config asks the fetch for the config's own
    # top (tests/test_gfs_fetch_follows_model_top.py), so the default
    # stands for GFS too.
    assert domain_wizard.emitted_model_top_pa("gfs") == DEFAULT_PA
    # 20CRv3's PSL series stops at 100 hPa and its fetch takes no top:
    # its emission stays inside the column it has.
    assert domain_wizard.emitted_model_top_pa("20crv3-cf") == 10000.0
    # No source at all (library callers) gets the default.
    assert domain_wizard.emitted_model_top_pa(None) == DEFAULT_PA


def _mapping_top_pa(profile_id: str) -> float | None:
    """The smallest pressure (Pa) a packaged mapping's ladder serves."""
    from woof.source_authorities import packaged_authorities
    mapping = json.loads(
        packaged_authorities(profile_id)["mapping"].read_text(
            encoding="utf-8"))
    vertical = mapping.get("coordinates", {}).get("vertical", {})
    if vertical.get("kind") == "model_level":
        measured = json.loads((Path(__file__).parent / "fixtures" /
            "model-level-source-tops.json").read_text(encoding="utf-8"))["profiles"]
        if profile_id not in measured:
            # A generated atmosphere can splice into a measured donor's
            # full pressure column. Require the pinned contributing mapping,
            # the identical vertical coordinate and pressure selector, and a
            # composition that actually supplies air_pressure from that donor.
            from woof.source_authorities import packaged_contributing_mappings
            authorities = packaged_authorities(profile_id)
            donors = packaged_contributing_mappings(profile_id)
            composition = json.loads(authorities["composition"].read_text())
            inherited = []
            for binding in composition.get("field_sources", {}).values():
                if "air_pressure" not in binding.get("fields", ()):
                    continue
                donor = json.loads(donors[binding["mapping_role"]].read_text())
                for candidate in measured:
                    reference = json.loads(packaged_authorities(candidate)["mapping"].read_text())
                    if (donor["coordinates"]["vertical"] == reference["coordinates"]["vertical"]
                            and donor["fields"].get("air_pressure") == reference["fields"].get("air_pressure")
                            and vertical.get("model_top_pressure_pa") == reference["coordinates"]["vertical"].get("model_top_pressure_pa")):
                        inherited.append(candidate)
            assert len(set(inherited)) == 1, f"{profile_id}: no unique measured pressure-column donor"
            profile_id = inherited[0]
        assert profile_id in measured, f"{profile_id}: model-level pressure coverage needs a measured top"
        row = measured[profile_id]
        if "model_top_pressure_pa" in vertical:
            # 69a0baec1's native HRRR and RAP mappings: a hybrid coordinate
            # that is pure pressure at its top, so the measured top-level
            # pressure of real objects is the bound itself, with no
            # height-to-pressure climatology between them.
            assert row["coverage_rule"] == (
                "measured top-level pressure of a pressure-based hybrid coordinate")
            assert row["declared_model_top_pressure_pa"] == vertical["model_top_pressure_pa"]
            assert float(row["measured_maximum_pa"]) >= float(vertical["model_top_pressure_pa"])
            return float(row["measured_maximum_pa"])
        reference = row["warm_season_reference"]
        assert reference["coverage_rule"] == "lower altitude bracket pressure"
        return max(float(row["measured_maximum_pa"]), max(reference["pressure_pa"]))
    if str(vertical.get("kind", "")) != "pressure":
        # Coefficient-based coordinates need their own pressure coverage evidence.
        return None
    levels = vertical.get("levels")
    if not levels:
        return None
    top = float(min(float(level) for level in levels))
    units = str(vertical.get("units", "Pa")).strip().lower()
    if units in ("pa", "pascal", "pascals"):
        return top
    if units in ("hpa", "mb", "mbar", "millibar", "millibars"):
        return top * 100.0
    raise AssertionError(
        f"{profile_id}: unrecognized vertical units {units!r}; teach this "
        "test the conversion before trusting the bound")


def test_every_planable_emission_sits_inside_its_source_column():
    """A bare wizard emission must be preparable from its own source.

    The concrete breakage this gate prevents: `woof domain --source X`
    followed by the emitted [fetch] block and `woof prep` refusing with
    "source top ... does not cover model top ..." -- a default that costs
    the user the whole acquisition before telling them it never could
    have worked.
    """
    checked = 0
    for source_id in wizard_planable_source_ids():
        adapter = get_source_adapter(source_id)
        if adapter.packaged_profile is None:
            continue
        top_pa = _mapping_top_pa(adapter.packaged_profile)
        if top_pa is None:
            continue
        emitted = domain_wizard.emitted_model_top_pa(source_id)
        assert emitted >= top_pa, (
            f"{source_id}: emitted p_top {emitted} Pa is above its "
            f"source's certified top {top_pa} Pa; declare "
            f"certified_source_top_pa on its registry row")
        checked += 1
    assert checked >= 5, "the gate found almost nothing to check"


def test_initialize_real_parameter_default():
    from woof.ingest.real import initialize_real
    parameter = inspect.signature(initialize_real).parameters["p_top"]
    assert parameter.default == DEFAULT_PA


def test_rrtmgp_workspace_field_default():
    import dataclasses
    from woof.core.model import SharedRRTMGPChunkWorkspace
    fields = {f.name: f
              for f in dataclasses.fields(SharedRRTMGPChunkWorkspace)}
    assert fields["p_top"].default == DEFAULT_PA


@pytest.mark.parametrize("function_name", (
    "rrtmgp_workspace_phases",
    "rrtmgp_workspace_shapes",
    "rrtmgp_column_shapes",
    "estimate_domain",
    "_workspace_total_bytes",
))
def test_preflight_parameter_defaults(function_name):
    from woof.core import preflight
    function = getattr(preflight, function_name)
    parameter = inspect.signature(function).parameters["p_top"]
    assert parameter.default == DEFAULT_PA

