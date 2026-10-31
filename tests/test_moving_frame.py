"""Moving-frame conventions, immutable forcing, and configuration doors."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config
from woof.moving_frame import (
    ground_relative_winds, initialize_moving_frame, moving_frame_attrs,
    validate_moving_frame, validate_uniform_surface,
)


def config(**values):
    return RunConfig(nx=16, ny=16, nz=4, dx=1000., dy=1000., ztop=10000.,
                     dt=1., run_seconds=3., **values)


def test_zero_frame_does_not_touch_signed_zero_or_object_graph():
    state = SimpleNamespace(u=np.array([-0., 1.], np.float32),
                            v=np.array([-0., -1.], np.float32))
    before = {key: value.tobytes() for key, value in vars(state).items()}
    initialize_moving_frame(state, config())
    assert before == {key: value.tobytes() for key, value in vars(state).items()}
    assert moving_frame_attrs(state) == {}


def test_init_once_output_ground_and_offsets_do_not_mutate_prognostics():
    cfg = config(umove=12., vmove=-8.)
    validate_run_config(cfg)
    state = SimpleNamespace(u=np.full(5, 12., np.float32),
                            v=np.full(5, -8., np.float32),
                            ht=np.zeros((1, 1)), rotational=False,
                            elapsed_seconds=3.)
    initialize_moving_frame(state, cfg)
    initialize_moving_frame(state, cfg)
    assert np.all(state.u == 0.) and np.all(state.v == 0.)
    u, v = ground_relative_winds(state)
    assert np.all(u == 12.) and np.all(v == -8.)
    assert np.all(state.u == 0.) and np.all(state.v == 0.)
    attrs = moving_frame_attrs(state)
    assert attrs["DOMAIN_X_OFFSET_M"] == 36.
    assert attrs["DOMAIN_Y_OFFSET_M"] == -24.
    with pytest.raises(ValueError, match="cannot change"):
        initialize_moving_frame(state, replace(cfg, umove=1.))


@pytest.mark.parametrize("values,reason", [
    ({"terrain_opt": 1}, "terrain"), ({"nested": True}, "nested"),
    ({"map_proj": 1}, "geostrophic"),
    ({"sf_surface_physics": 2}, "ground-fixed"),
    ({"ifire": 2}, "ground-fixed"),
])
def test_scope_refusals_name_the_untranslated_ground_or_balance(values, reason):
    with pytest.raises(ValueError, match=reason):
        validate_moving_frame(config(umove=12., **values))


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "12"])
def test_speed_must_be_finite_numeric(value):
    with pytest.raises(ValueError, match="finite speed"):
        validate_moving_frame(config(vmove=value))


def test_uniform_prescribed_surface_and_direct_coriolis_refusal():
    validate_uniform_surface(tsk=np.full((2, 2), 300.))
    with pytest.raises(ValueError, match="translation"):
        validate_uniform_surface(tsk=np.array([299., 300.]))
    state = SimpleNamespace(rotational=True)
    with pytest.raises(ValueError, match="geostrophic"):
        initialize_moving_frame(state, config(umove=1.))


def test_default_identity_and_active_frame_binding():
    from woof.io.restart import configuration_echo
    assert "umove" not in configuration_echo(config())
    assert "vmove" not in configuration_echo(config())
    assert configuration_echo(config(umove=12., vmove=-8.))["umove"] == 12.


def test_toml_door_and_keyword_only_constructor(tmp_path):
    import inspect
    from woof.config import load_config
    path = tmp_path / "frame.toml"
    path.write_text("[grid]\nnx=16\nny=16\nnz=4\ndx=1000.0\ndy=1000.0\nztop=10000.0\n"
                    "[dynamics]\ndt=1.0\numove=12.0\nvmove=-8.0\n"
                    "[run]\nrun_seconds=3.0\n", encoding="utf-8")
    cfg = load_config(path)
    assert (cfg.umove, cfg.vmove) == (12., -8.)
    for key in ("umove", "vmove"):
        assert inspect.signature(RunConfig).parameters[key].kind is inspect.Parameter.KEYWORD_ONLY


def test_experiment_shared_frame_and_tree_refusal():
    from datetime import datetime
    from woof.experiment import build_experiment
    raw = {"experiment": {"name": "translation", "start_time": datetime(2000, 1, 1),
                           "run_seconds": 60., "restart_interval_s": 0.},
           "shared": {"nz": 4, "ztop": 10000., "umove": 12., "vmove": -8.},
           "domain": [{"grid_id": 1, "parent_id": 0, "i_parent_start": 1,
                       "j_parent_start": 1, "parent_grid_ratio": 1,
                       "parent_time_step_ratio": 1, "nx": 64, "ny": 64,
                       "dx": 1000., "time_step": 1, "history_interval_s": 60.}]}
    exp = build_experiment(raw, "translation")
    assert (exp.domains[0].run.umove, exp.domains[0].run.vmove) == (12., -8.)
    raw["domain"].append({"grid_id": 2, "parent_id": 1, "i_parent_start": 20,
                           "j_parent_start": 20, "parent_grid_ratio": 3,
                           "parent_time_step_ratio": 3, "nx": 30, "ny": 30,
                           "history_interval_s": 60.})
    with pytest.raises(ValueError, match="nested|nests"):
        build_experiment(raw, "translation")


def test_streamed_reuse_refuses_double_subtraction_before_building():
    from woof.core.streaming import attach, make_stepper, StreamingRefused
    cfg = config(umove=12.)
    with pytest.raises(StreamingRefused, match="subtract translation again"):
        make_stepper(None, cfg, decision=SimpleNamespace(stream=True))
    with pytest.raises(StreamingRefused, match="subtract translation again"):
        attach(None, cfg, None, tile_state_factory=None)


def test_direct_rotation_setter_refuses_before_mutating_arrays():
    from woof.core.state import DomainState
    state = DomainState(config(umove=12.), array_module=np)
    with pytest.raises(ValueError, match="geostrophic"):
        state.set_map_coriolis(f=np.full((16, 16), 1.e-4))
    assert np.all(state.f == 0.)


def test_direct_nest_attachment_refuses_untranslated_ground_geometry():
    from woof.ingest.lateral_bc import attach_nest_boundaries
    with pytest.raises(ValueError, match="nested ground geometry translated"):
        attach_nest_boundaries(SimpleNamespace(_moving_frame_velocity=(12., -8.)),
                               {}, clock=None)


def test_namelist_zero_is_inert_and_real_data_translation_is_named(tmp_path):
    from test_namelist_import import _generic_hierarchy_pair
    from woof.namelist_import import import_namelists
    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    original = inp.read_text()
    baseline, _ = import_namelists(wps, inp, name="translation")
    inp.write_text(original.replace("&dynamics", "&dynamics\n umove=0., vmove=0.,"))
    zero, _ = import_namelists(wps, inp, name="translation")
    assert zero == baseline
    inp.write_text(original.replace("&dynamics", "&dynamics\n vmove=-8.,"))
    with pytest.raises(ValueError, match="ground.*translation"):
        import_namelists(wps, inp, name="translation")


def test_coupled_boundary_shift_includes_mass_tendency_on_every_side():
    from woof.ingest.lateral_bc import (
        build_lateral_boundaries, lateral_boundaries_in_frame,
    )
    nz, ny, nx = 4, 16, 16
    c1 = np.linspace(1., .2, nz)
    c2 = np.linspace(0., 10., nz)
    state = SimpleNamespace(_moving_frame_velocity=(12., -8.), mub=100.,
                            has_msf=False, c1h=c1, c2h=c2)

    def snapshot(time):
        mu = np.arange(ny * nx).reshape(ny, nx) + time * 3.
        mux = np.concatenate((mu[:, :1], .5 * (mu[:, :-1] + mu[:, 1:]), mu[:, -1:]), axis=1)
        muy = np.concatenate((mu[:1], .5 * (mu[:-1] + mu[1:]), mu[-1:]), axis=0)
        return {"mu": mu[None],
                "u": 12. * (c1[:, None, None] * (100. + mux) + c2[:, None, None]),
                "v": -8. * (c1[:, None, None] * (100. + muy) + c2[:, None, None])}

    original = build_lateral_boundaries([snapshot(0.), snapshot(2.)], [0., 2.])
    shifted = lateral_boundaries_in_frame(state, original)
    assert shifted is lateral_boundaries_in_frame(state, shifted)
    for name in ("u", "v"):
        for side in ("west", "east", "south", "north"):
            wind = getattr(shifted.intervals[0].fields[name], side)
            np.testing.assert_allclose(wind.value, 0., atol=1.e-12)
            np.testing.assert_allclose(wind.tendency, 0., atol=1.e-12)
            assert not wind.value.flags.writeable
            assert np.any(getattr(original.intervals[0].fields[name], side).value != 0)
