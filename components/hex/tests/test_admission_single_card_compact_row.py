"""The MPAS rank gate prices one context on one card with the first gate's qualified compact row.

THE BREAKAGE THIS PREVENTS (global16 hex-fit16, 2026-11-06, woof hex 0.4.13 and 0.4.14): the second MPAS admission
gate (mpasphys.memory_admission.admit) priced a single-card run with the legacy per-card row, so the 60 km globe
(x1.163842, 55 levels, MPAS convection_permitting) that the first gate had just admitted from the RTX 5070 Ti compact
row (15,623,782,400 bytes) was refused on that 16 GB card as "card 0 needs 29,930,042,163 bytes"; the whole 24 h run
peaks 11.1 GiB live and 14,492 MiB on the card.  Rank layouts (--samehost-devices) keep their measured rank rows.
"""
import contextlib
import json
import sys
import types
from types import SimpleNamespace

import pytest

from woof.hex import device_admission as da
from woof.hex.mpasphys import memory_admission as ma

CARD16 = da.KNOWN_CARDS["16gib-70sm"]
CELLS60, LEVELS = 163_842, 55
FREE16 = 16_368_795_648      # what an RTX 5070 Ti 16 GB admission recorded free, 2026-11-06
SIXTEEN_GB = 16_000_000_000
R8_CELLS = [5927, 5935, 5952, 5947, 6004, 5882, 5931, 5913]   # x1.40962 seed-841 part.8, halos included


@pytest.fixture(autouse=True)
def _compact_defaults(monkeypatch):
    from woof.hex import numerics
    monkeypatch.setattr(numerics, "_mode", "fast")
    for key in ("WOOF_HEX_MEMBER_MEMORY", "WOOF_HEX_THOMPSON_WORKSPACE", "WOOF_HEX_THOMPSON_TILE_COLUMNS",
                "WOOF_HEX_THOMPSON_WORKSPACE_MIB", "WOOF_HEX_TRANSPORT_WORKSPACE_BYTES",
                "WOOF_HEX_LIMITER_SPECIES_PER_PASS", "GPUWM_HEX_POOL_RELEASE_MIN_BYTES",
                "GPUWM_HEX_RELEASE_HEADROOM", "WOOF_HEX_POOL_RELEASE", "GPUWM_HEX_LMEM_RELEASE"):
        monkeypatch.delenv(key, raising=False)


def _gate1_price(cells=CELLS60, suite="convection_permitting"):
    model = da.model_for_run(CARD16, "global", LEVELS, route="mpas", suite=suite, cells=cells)
    return da.route_required_free_bytes(cells, model, route="mpas")


def test_60km_single_card_is_priced_with_the_first_gates_row_below_16_gb():
    need = ma.rank_requirements([CELLS60], [0], {0: CARD16}, LEVELS, single_context=True,
                                suite="convection_permitting")[0]
    assert need == _gate1_price()
    assert need < SIXTEEN_GB and need < FREE16
    legacy = ma.rank_requirements([CELLS60], [0], {0: CARD16}, LEVELS)[0]
    assert legacy > FREE16          # the old price, which refused the run


def test_a_footprint_option_keeps_the_legacy_row():
    legacy = ma.rank_requirements([CELLS60], [0], {0: CARD16}, LEVELS)[0]
    assert ma.rank_requirements([CELLS60], [0], {0: CARD16}, LEVELS, single_context=True,
                                suite="convection_permitting", qualified=False)[0] == legacy
    # Outside the measured column range there is no qualified row: legacy, as before.
    assert ma.rank_requirements([40_000], [0], {0: CARD16}, LEVELS, single_context=True,
                                suite="convection_permitting")[0] == \
        ma.rank_requirements([40_000], [0], {0: CARD16}, LEVELS)[0]


def test_rank_layouts_price_as_before():
    # The 8-rank x1.40962 co-tenant row (hex 0.4.14) and one rank per card are untouched.
    for key in ("32gib-170sm", "96gib-188sm", "16gib-70sm"):
        card = da.KNOWN_CARDS[key]
        assert ma.rank_requirements(R8_CELLS, [0] * 8, {0: card}, 59)[0] == \
            ma.cotenant_card_bytes(R8_CELLS, 59, da.model_for_card(card, "global", levels=59))[0]
    assert ma.rank_requirements(R8_CELLS, [0] * 8, {0: da.KNOWN_CARDS["32gib-170sm"]}, 59)[0] == 32_247_696_589


def _door(monkeypatch, tmp_path, **options):
    """Run the real rank gate on a 16 GB card seam (no GPU, no init decode)."""
    from woof.hex import forecast_door, mesh_rows
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(cuda=types.SimpleNamespace(
        Device=lambda device=0: contextlib.nullcontext(), runtime=types.SimpleNamespace(is_hip=False))))
    monkeypatch.setattr(mesh_rows, "_dimensions", lambda path: {"nCells": CELLS60, "nVertLevels": LEVELS})
    monkeypatch.setattr(forecast_door, "read_card_profile", lambda device=None: CARD16)
    monkeypatch.setattr(forecast_door, "measure_device_memory", lambda: (FREE16, 16_303 * da.MIB))
    arguments = SimpleNamespace(init=str(tmp_path / "init.nc"), out=str(tmp_path / "out"), samehost_devices=None,
                                lbc_dir=None, physics=None, physics_profile=None, physics_namelist=None,
                                physics_default_backend=None, local_timestep=False, fullstep_local_timestep=False,
                                stop_on_refusal=False, physics_suite=None,
                                native_admission_suite="convection_permitting")
    for key, value in options.items():
        setattr(arguments, key, value)
    return arguments


def test_the_rank_gate_admits_the_60km_globe_on_a_16_gb_card(monkeypatch, tmp_path):
    record = ma.admit(_door(monkeypatch, tmp_path))
    assert record["required_bytes"][0] == _gate1_price() < SIXTEEN_GB
    assert record["card_pricing"][0]["row"] == "qualified compact row"
    written = json.loads((tmp_path / "out" / "device-memory-admission.json").read_text())
    assert written["required_bytes"]["0"] == record["required_bytes"][0]


def test_the_rank_gate_still_refuses_with_a_footprint_option(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="card 0 needs"):
        ma.admit(_door(monkeypatch, tmp_path, physics_namelist="namelist.atmosphere"))
