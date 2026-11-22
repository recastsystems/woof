"""A microphysics picked over an ArWen-route suite stops by name when the engine has no cumulus contract for it.

THE BREAKAGE (hex 0.5.0, 2026-11-10): TEMPO over arwen_mpas_column on a zoom globe (Grell-Freitas by the
resolution rule) on public gpuwm 2.9.13 stopped inside the engine with "KF has no verified phase-output
contract for mp_physics=900" and no word on what to do.
"""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from woof.hex.drivers import run_cuda_v841_forecast as driver
from woof.hex.errors import ConfigurationRefusal
from woof.hex.physics_suites import mpas_suite_names
from woof.hex.species_row import species_row_for_scheme


def _engine(monkeypatch, admits):
    seen = []

    def contract(cfg):
        seen.append((cfg.cu_physics, cfg.mp_physics))
        if cfg.mp_physics not in admits:
            raise ValueError(f"KF has no verified phase-output contract for mp_physics={cfg.mp_physics}")
        return ("rqr",)

    module = ModuleType("woof.core.physics")
    module._cumulus_optional_tendency_components = contract
    monkeypatch.setitem(sys.modules, "woof.core.physics", module)
    return seen


def test_tempo_over_grell_freitas_without_the_engine_row_stops_by_name(monkeypatch):
    seen = _engine(monkeypatch, admits={8})
    physics = SimpleNamespace(suite="arwen_mpas_column")
    with pytest.raises(ConfigurationRefusal, match="Update woof"):
        driver.refuse_pick_without_cumulus_contract(physics, species_row_for_scheme("mp_tempo"), "cu_grell_freitas")
    assert seen == [(3, 900)]


def test_the_suites_own_microphysics_with_grell_freitas_runs(monkeypatch):
    _engine(monkeypatch, admits={8})
    physics = SimpleNamespace(suite="arwen_mpas_column")
    driver.refuse_pick_without_cumulus_contract(physics, species_row_for_scheme("mp_thompson"), "cu_grell_freitas")


def test_the_stop_retires_on_an_engine_that_carries_the_row(monkeypatch):
    _engine(monkeypatch, admits={8, 900})
    physics = SimpleNamespace(suite="arwen_mpas_column")
    driver.refuse_pick_without_cumulus_contract(physics, species_row_for_scheme("mp_tempo"), "cu_grell_freitas")


def test_no_convection_and_mpas_suites_are_not_asked(monkeypatch):
    from woof.hex.convection_admission import SCHEME_OFF

    seen = _engine(monkeypatch, admits=set())
    row = species_row_for_scheme("mp_tempo")
    driver.refuse_pick_without_cumulus_contract(SimpleNamespace(suite="arwen_mpas_column"), row, SCHEME_OFF)
    driver.refuse_pick_without_cumulus_contract(SimpleNamespace(suite=mpas_suite_names()[0]), row, "cu_grell_freitas")
    driver.refuse_pick_without_cumulus_contract(None, row, "cu_grell_freitas")
    assert seen == []
