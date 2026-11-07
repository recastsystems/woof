"""A column-suite run's configuration crosses the rank link to a rank process.

THE BREAKAGE (rrfs_v2_dev on four cards, hex 0.4.11, run_fb4825f5121040b2): the suite's configuration class is built
at run time, pickle finds a class by its module attribute, and every column-suite run on rank processes stopped with
PicklingError before its first step.  A one-card run sends no request, so nothing showed it there.
"""

import pickle
import subprocess
import sys
from dataclasses import replace

import pytest

from woof.hex import cuda_suite_physics_v841 as adapter
from woof.hex import licence_gate
from woof.hex.config_v841 import (
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
    V841UpperDampingSmagorinskyConfig,
)
from woof.hex.rank_processes import _dumps


def _suite_values(config):
    return {name: getattr(config, name) for name in config.__dataclass_fields__}


@pytest.mark.parametrize("run_type", [V841MpasColumnPhysicsSmagorinskyGwdoConfig, V841UpperDampingSmagorinskyConfig])
def test_suite_config_round_trip_in_this_process(run_type):
    config = adapter.suite_config(run_type(), "mynn_ruc_gsl_column")
    config = replace(config, config_physics_suite="rrfs_v2_dev")
    back = pickle.loads(_dumps({"host": {"config": config}}, []))["host"]["config"]
    assert type(back) is type(config)
    assert _suite_values(back) == _suite_values(config)
    back.validate()


def test_suite_config_round_trip_in_a_fresh_process(tmp_path):
    """A rank process has built no suite class when its request arrives: the words decide, not this process's cache."""
    config = adapter.suite_config(V841MpasColumnPhysicsSmagorinskyGwdoConfig(), "mynn_ruc_gsl_column")
    path = tmp_path / "request.pickle"
    path.write_bytes(_dumps({"config": config}, []))
    code = (
        "import pickle, sys\n"
        "config = pickle.loads(open(sys.argv[1], 'rb').read())['config']\n"
        "config.validate()\n"
        "print(type(config).__name__, config.config_physics_suite_profile, config.config_pbl_scheme)\n"
    )
    done = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True)
    assert done.stdout.split() == [type(config).__name__, config.config_physics_suite_profile, config.config_pbl_scheme]


def test_generic_seam_selection_round_trip():
    """The generic seam's rows are registered at resolve time; a rank process replays the resolve after its request."""
    profile = dict(adapter.physics_suite("mynn-ruc-gsl"), id="generic-rank-link", backend_row="generic_rank_link_row")
    adapter.register_profile(profile)
    config = adapter.suite_config(V841MpasColumnPhysicsSmagorinskyGwdoConfig(), "generic_rank_link_row")
    back = pickle.loads(_dumps(config, []))
    assert back == config
    back.validate()


def test_licence_annotated_config_round_trip():
    """A run under a licence opt-in holds the same configuration through the annotated type."""
    config = V841MpasColumnPhysicsSmagorinskyGwdoConfig()
    values = {name: getattr(config, name) for name in config.__dataclass_fields__}
    annotated = licence_gate._annotated_type(type(config))(**values)
    object.__setattr__(annotated, "_licence_opt_in", ({"terms": "t", "slot": "s"},))
    back = pickle.loads(_dumps(annotated, []))
    assert type(back) is type(annotated)
    assert back.__dict__["_licence_opt_in"] == ({"terms": "t", "slot": "s"},)


@pytest.mark.parametrize("land", ["ruc", "noah", "none"])
def test_a_land_surface_other_than_noahmp_has_no_noahmp_routing_census(land):
    """KeyError 'sflx_land_columns' stopped every such run on one card and on four, from 0.4.5 on."""

    class Constructor:
        selection = {"land_surface_scheme": land}

    class Stub:
        _constructor = Constructor()

    assert adapter.PersistentSuiteCudaPhysicsBackendV841.noahmp_routing_census(Stub()) is None


def test_every_rank_holds_the_sea_ice_carriers_of_a_mesh_with_ocean_ice(monkeypatch):
    """A rank with no polar columns held none: "boundary entry paths diverge" at the end of a whole forecast."""
    import types

    import numpy as np

    from woof.hex.drivers.samehost_forecast_v841 import domain_ocean_ice

    assert domain_ocean_ice({"xice": np.array([0.0, 0.2])}) is True
    assert domain_ocean_ice({"xice": np.array([0.0, 0.0])}) is False
    assert domain_ocean_ice({"xice": np.array([0.0, 0.2]), "lakemask": np.array([0.0, 1.0])}) is False

    seeded = []
    seaice = types.ModuleType("woof.ocean.seaice")
    seaice.COLD_SEEDED = frozenset({"ice_volume", "ice_skin", "ice_thickness"})
    seaice.cold_seed_in_place = lambda fields, cfg, absent: seeded.append(sorted(absent))
    cupy = types.ModuleType("cupy")
    cupy.zeros_like = np.zeros_like
    monkeypatch.setitem(sys.modules, "woof", types.ModuleType("woof"))
    monkeypatch.setitem(sys.modules, "woof.ocean", types.ModuleType("woof.ocean"))
    monkeypatch.setitem(sys.modules, "woof.ocean.seaice", seaice)
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    def seam(fields, prognostic=True):
        cfg = types.SimpleNamespace(seaice_prognostic=prognostic, sf_surface_physics=1)
        return types.SimpleNamespace(_driver=types.SimpleNamespace(fields=fields), _cfg=cfg)

    fresh = seam({"tsk": np.ones(3, dtype=np.float32)})
    assert adapter.prime_rank_ocean_ice(fresh) == ["ice_skin", "ice_thickness", "ice_volume"]
    assert seeded == [["ice_skin", "ice_thickness", "ice_volume"]]
    assert fresh._driver.fields["ice_volume"].shape == (3,)
    # A rank with ice of its own was seeded by the engine; ice switched off stays off.
    assert adapter.prime_rank_ocean_ice(seam({"tsk": np.ones(3), "ice_volume": np.ones(3)})) == []
    assert adapter.prime_rank_ocean_ice(seam({"tsk": np.ones(3)}, prognostic=False)) == []
    assert len(seeded) == 1
