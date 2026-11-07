"""Host lifecycle checks for adaptive-habit moments, without a CUDA device."""
import numpy as np

from woof.core.state import DomainState
from woof.io import restart
from woof.physics_registry import consumer_row_for_selector
from test_ensemble_batch_state import config, pack, prepared


def test_restart_preserves_every_habit_word_and_precipitation_accumulator(tmp_path):
    cfg = config(moist=True, mp_physics=55, diff_opt=0)
    state = DomainState(cfg, array_module=np)
    row = consumer_row_for_selector("microphysics", "native_column", 55)
    words = np.array([0, 0x80000000, 1, 0x7FC00123, 0x3F800000], np.uint32)
    for index, name in enumerate(row["transport"]):
        array = getattr(state, name)
        array.view(np.uint32).flat[:] = np.resize(np.roll(words, index), array.size)
    from woof.core.physics_inventory import microphysics_scratch_slots
    slots = [slot for _, slot in microphysics_scratch_slots(55)]
    for index, slot in enumerate(slots):
        state.scratch((cfg.ny, cfg.nx), slot)[...] = np.float32(index + 1)
    state.elapsed_seconds = 12.0
    path = restart.write_restart(tmp_path / "adaptive-habit.npz", state, cfg)
    fresh = DomainState(cfg, array_module=np)
    info = restart.restore_restart(path, fresh, cfg)
    assert info.elapsed_seconds == fresh.elapsed_seconds == 12.0
    for name in row["transport"]:
        assert getattr(fresh, name).tobytes() == getattr(state, name).tobytes(), name
    for slot in slots:
        assert fresh.existing_scratch(slot).tobytes() == state.existing_scratch(slot).tobytes()


def _members():
    cfg = config(moist=True, mp_physics=55)
    row = consumer_row_for_selector("microphysics", "native_column", 55)
    members = (prepared(cfg), prepared(cfg))
    for member_index, member in enumerate(members):
        for field_index, name in enumerate(row["transport"]):
            member.arrays[name].fill(100 * member_index + field_index + 1)
    return cfg, row, members


def test_member_packing_preserves_habit_moments_and_independent_ownership():
    _, row, members = _members()
    batch = pack(members)
    assert batch._registered_mass_slots == ("qi", "qi2", "qi3")
    for name in row["transport"]:
        for member_index, member in enumerate(members):
            actual = batch.member_view(name, member_index)
            assert actual.tobytes() == member.arrays[name].tobytes()
            assert not np.shares_memory(actual, member.arrays[name])
        assert not np.shares_memory(batch.member_view(name, 0), batch.member_view(name, 1))


def test_streamed_history_keeps_hybrid_mass_coefficients_for_three_habit_ice_path():
    from tilestream.test_store_direct_frame import domain_setup
    cfg = config(moist=True, mp_physics=55)
    setup = domain_setup(cfg)
    statics = setup.output_statics(cfg.nz, cfg.ny, cfg.nx)
    assert statics["C1H"].tobytes() == setup.c1h.tobytes()
    assert statics["C2H"].tobytes() == setup.c2h.tobytes()
    other = domain_setup(config(moist=True, mp_physics=8))
    assert not {"C1H", "C2H"} & other.output_statics(cfg.nz, cfg.ny, cfg.nx).keys()


def test_batched_acoustic_and_buoyancy_bind_all_three_ice_masses(monkeypatch):
    from woof.ensemble import batch_acoustic, batch_bigstep
    cfg, _, members = _members()
    batch = pack(members, scratch_slots={name: np.float32 for name in
                 ("acoustic_cqu", "acoustic_cqv", "acoustic_cqw")})
    calls = {}

    class Bindings:
        def __init__(self, state, config):
            assert state is batch and config is cfg

        def bind(self, entry, args, pointers, grid):
            calls[entry] = (args, dict(pointers))
            return lambda: None

    monkeypatch.setattr(batch_acoustic, "_Bindings", Bindings)
    batch_acoustic.prepare_moist_cq(batch, cfg)
    args, pointers = calls["calc_cq"]
    assert int(args[10]) == 6
    for parameter, carrier in zip(("qi", "qs", "qg"), ("qi", "qi2", "qi3"), strict=True):
        assert pointers[parameter] is getattr(batch, carrier)

    def bound(state, entry, args, grid, *, bindings):
        assert state is batch
        calls[entry] = (args, bindings)
        return lambda: None

    monkeypatch.setattr(batch_bigstep, "_bound_launch", bound)
    batch_bigstep.prepare_slow_buoyancy(batch, cfg)
    args, bindings = calls["slow_buoyancy"]
    assert bindings["qi"] == "qi"
    assert bindings["qs"] == "qi2"
    assert bindings["qg"] == "qi3"
    assert args[8] is batch.qi and args[9] is batch.qi2 and args[10] is batch.qi3
