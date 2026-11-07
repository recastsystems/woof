"""Column word copies and default physics leaves versus ordinary launchers."""

import numpy as np
import pytest

from conftest import requires_gpu
from woof.ensemble.batch_physics import (ColumnBuffers, ColumnField, YSU_COLUMNS,
    YSU_SURFACES, prepare_ysu_column_batch, prepare_sfclay_column_batch,
    prepare_noah_column_batch, SFCLAY_INPUTS)

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("members", [1, 4, 10, 20])
@pytest.mark.parametrize("dtype", ["float32", "int32", "uint32"])
def test_column_pack_unpack_preserves_every_word(members, dtype):
    import cupy as cp
    rng = np.random.default_rng(218)
    words = rng.integers(0, 2**32, (members, 7, 5, 9), dtype=np.uint32)
    values = cp.asarray(words.view(dtype))
    fields = ColumnBuffers((ColumnField("field", (7, 5, 9), dtype),), members=members,
                           available_bytes=1 << 24)
    packed = fields.pack("field", values)
    expected = words.transpose(1, 0, 2, 3).reshape(7, members * 5, 9)
    assert packed.get().view(np.uint32).tobytes() == expected.tobytes()
    out = cp.empty_like(values)
    fields.unpack("field", out)
    assert out.get().view(np.uint32).tobytes() == words.tobytes()
    assert fields.converted_bytes == 4 * values.nbytes
    with pytest.raises(ValueError, match="overlap"):
        fields.pack("field", fields.storage.arrays["physics:field"])


def test_member_ring_mask_keeps_each_stripe_separate():
    import cupy as cp
    values = np.ones((4, 3, 5, 9), np.float32)
    fields = ColumnBuffers((ColumnField("field", (3, 5, 9)),), members=4, available_bytes=1 << 20)
    fields.pack("field", cp.asarray(values))
    fields.mask_member_rings("field")
    out = cp.empty(values.shape, cp.float32)
    fields.unpack("field", out)
    values[:, :, (0, -1), :] = 0
    values[:, :, :, (0, -1)] = 0
    assert out.get().tobytes() == values.tobytes()


@pytest.mark.parametrize("members,width", [(1, 1), (4, 1), (10, 2), (20, 2)])
def test_compact_physics_ring_restores_each_member_and_pins_heating(members, width):
    import cupy as cp
    from woof.ensemble.batch_physics import PreparedPhysicsRing
    rng = np.random.default_rng(538)
    words = rng.integers(0, 2**32, (7, members * 8, 9), np.uint32)
    original = cp.asarray(words.view(np.float32))
    heating = cp.ones_like(original)
    ring = PreparedPhysicsRing({"field": original, "heating": heating},
                               members=members, ny=8, nx=9, width=width,
                               zero_fields=("heating",), available_bytes=1 << 24)
    ring.capture()
    original.fill(np.float32(123))
    heating.fill(np.float32(456))
    ring.restore()
    mask = np.zeros((8, 9), bool)
    mask[:width] = mask[-width:] = True
    mask[:, :width] = mask[:, -width:] = True
    mask = np.broadcast_to(np.tile(mask, (members, 1))[None], words.shape)
    expected = np.where(mask, words, np.float32(123).view(np.uint32)).astype(np.uint32)
    assert original.get().view(np.uint32).tobytes() == expected.tobytes()
    np.testing.assert_array_equal(heating.get(), np.where(mask, 0, 456))


@pytest.mark.parametrize("members", [1, 4, 10, 20])
@pytest.mark.parametrize("shared", [False, True])
def test_packed_ysu_one_launch_matches_each_original_member(members, shared):
    import cupy as cp
    from test_ysu import _batched_columns
    from woof.core.ysu import launch_ysu
    hosts = [_batched_columns(nz=12, seed=991 + m) for m in range(members)]
    hosts = [{**host, "rthraten": np.zeros_like(host["theta"])} for host in hosts]
    declared = tuple(ColumnField(name, hosts[0][name].shape) for name in YSU_COLUMNS + YSU_SURFACES if not (shared and name == "xland"))
    buffers = ColumnBuffers(declared, members=members, available_bytes=1 << 27)
    arrays = {}
    for name in YSU_COLUMNS + YSU_SURFACES:
        if shared and name == "xland":
            arrays[name] = cp.asarray(hosts[0][name])
        else:
            arrays[name] = buffers.pack(name, cp.asarray(np.stack([h[name] for h in hosts])))
    bound = prepare_ysu_column_batch(arrays, members=members, ny=2, nx=5, dt=45,
                                     available_bytes=1 << 27, shared_fields=("xland",) if shared else ())
    before = {name: a.get().tobytes() for name, a in arrays.items()}
    output = bound()
    for member, host in enumerate(hosts):
        ordinary = launch_ysu(**{name: cp.asarray(host[name]) for name in YSU_COLUMNS + YSU_SURFACES}, dt=45)
        for name, array in ordinary.items():
            got = output[name][:, member * 2:(member + 1) * 2, :] if array.ndim == 3 else output[name][member * 2:(member + 1) * 2, :]
            assert got.get().tobytes() == array.get().tobytes(), (member, name, shared)
    assert {name: a.get().tobytes() for name, a in arrays.items()} == before


@pytest.mark.parametrize("members", [1, 4, 10, 20])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("option", [1, 91])
def test_mm5_one_launch_matches_each_original_member(members, shared, option):
    import cupy as cp
    from test_sfclay import _surface_cases
    from woof.core.sfclay import sfclay
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    hosts = [_surface_cases(seed=223 + m, ny=3, nx=5) for m in range(members)]
    inputs = {name: cp.asarray(hosts[0][name]) if shared and name in ("xland", "lakemask")
              else cp.asarray(np.concatenate([h[name] for h in hosts], axis=0)) for name in SFCLAY_INPUTS}
    outputs = {name: cp.zeros((members * 3, 5), cp.float32) for name in SFCLAY_OUTPUTS}
    for name in ("znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol"):
        outputs[name].set(np.concatenate([h.get(name, np.zeros((3, 5), np.float32)) for h in hosts], axis=0))
    bound = prepare_sfclay_column_batch(inputs, outputs, members=members, ny=3, nx=5,
                                        option=option, dx=1000,
                                        shared_fields=("xland", "lakemask") if shared else ())
    before = {name: a.get().tobytes() for name, a in inputs.items()}
    bound()
    for member, host in enumerate(hosts):
        ordinary = sfclay(**{name: cp.asarray(value) for name, value in host.items()}, option=option, dx=1000)
        for name in SFCLAY_OUTPUTS:
            assert outputs[name][member * 3:(member + 1) * 3].get().tobytes() == getattr(ordinary, name).get().tobytes(), (member, name, shared, option)
    assert {name: a.get().tobytes() for name, a in inputs.items()} == before


@pytest.mark.parametrize("members", [1, 4, 10, 20])
@pytest.mark.parametrize("shared", [False, True])
def test_noah_one_launch_matches_each_original_member(members, shared):
    import cupy as cp
    from test_noah import _base_col, _grid_from_cols, DZS
    from woof.core.noah import _F2D, _F3D, launch_noah, load_tables, pack_params
    params = pack_params(load_tables())
    hosts = []
    for member in range(members):
        columns = [_base_col(params, rng=np.random.default_rng(771 + member * 10 + col), rain=0.3 * member) for col in range(5)]
        fields, _, ny, nx = _grid_from_cols(columns)
        for name in ("smstav", "smstot", "noahres", "reslin", "chklowq"):
            fields[name] = np.zeros((ny, nx), np.float32)
        fields["smcrel"] = np.zeros((4, ny, nx), np.float32)
        fields["ebal"] = np.zeros((ny, nx), np.int32)
        hosts.append({name: np.asarray(value, dtype=np.int32 if name in ("ivgtyp", "isltyp", "ebal") else np.float32) for name, value in fields.items()})
    ny, nx = hosts[0]["tsk"].shape
    shared_names = ("ivgtyp", "isltyp", "xland", "shdmin", "shdmax", "tmn", "snoalb", "embck") if shared else ()
    fields = {}
    for name in set(_F2D + _F3D) | {"ivgtyp", "isltyp", "ebal"}:
        if name in shared_names:
            fields[name] = cp.asarray(hosts[0][name])
        else:
            fields[name] = cp.asarray(np.concatenate([h[name] for h in hosts], axis=1 if name in _F3D else 0))
    bound = prepare_noah_column_batch(fields, params, members=members, ny=ny, nx=nx,
                                     dt=12, dzs=DZS, shared_fields=shared_names)
    bound(1)
    for member, host in enumerate(hosts):
        ordinary = {name: cp.asarray(value) for name, value in host.items()}
        launch_noah(ordinary, params, 12, DZS, itimestep=1)
        for name in fields:
            got = fields[name] if name in shared_names else fields[name][:, member * ny:(member + 1) * ny] if name in _F3D else fields[name][member * ny:(member + 1) * ny]
            assert got.get().tobytes() == ordinary[name].get().tobytes(), (member, name, shared)


@pytest.mark.parametrize("members", [1, 4, 10])
@pytest.mark.parametrize("terrain", [False, True])
@pytest.mark.parametrize("use_theta_m", [0, 1])
def test_complete_thompson_column_chain_and_member_rings_match_stock(members, terrain, use_theta_m):
    import cupy as cp
    from test_mp_spec_zone_ring import _square_moist_state
    from woof.core import microphysics
    from woof.core.physics_inventory import ring_guard_row
    from woof.ensemble.batch_physics import (PackedThompsonState,
        prepare_thompson_column_batch, thompson_column_scratch_fields)
    ny, nx, nz = 8, 9, 12
    states = []
    for member in range(members):
        state, cfg = _square_moist_state(ny=ny, nx=nx, nz=nz, mp=8, dt=12, specified=True)
        state.thp += np.float32(member * 0.01)
        if terrain:
            patch = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) * np.float32(0.5)
            state.phb = cp.asarray(cp.asnumpy(state.phb)[:, None, None] + patch[None] * np.float32(9.81))
            state.thb = cp.asarray(cp.asnumpy(state.thb)[:, None, None] + patch[None] * np.float32(0.001))
        if use_theta_m:
            from woof.core import constants as c
            state.thp[...] = (state.total_theta()
                              * (np.float32(1.0) + np.float32(c.RVOVRD) * state.qv)
                              - (state.thb if state.thb.ndim == 3 else state.thb[:, None, None]))
            state.use_theta_m = 1
        states.append(state)
    from dataclasses import replace
    cfg = replace(cfg, use_theta_m=use_theta_m)
    names = set(ring_guard_row(8)["state_fields"]) | {
        "p", "thp", "qv", "qc", "qr", "qi", "ni", "nr", "qs", "qg",
        "h_diabatic", "effc", "effi", "effs", "php", "w"}
    names = tuple(sorted(name for name in names if getattr(states[0], name, None) is not None))
    declared = tuple(ColumnField(name, getattr(states[0], name).shape) for name in names)
    staged = ColumnBuffers(declared, members=members, available_bytes=1 << 29)
    arrays = {}
    for name in names:
        arrays[name] = staged.pack(name, cp.stack([getattr(state, name) for state in states]))
    arrays.update(thb=states[0].thb, phb=states[0].phb, use_theta_m=use_theta_m)
    # The production admission (batch_physics_init) passes the run's
    # correct-mode WB1 key; so does this fixture.
    scratch_fields = thompson_column_scratch_fields(
        nz=nz, ny=ny, nx=nx, cloud_booking=cfg.thompson_cloud_surface_booking,
        use_theta_m=use_theta_m)
    scratch = ColumnBuffers(scratch_fields, members=members, available_bytes=1 << 29)
    packed = PackedThompsonState(arrays, {field.name: scratch.packed(field.name) for field in scratch_fields})
    bound = prepare_thompson_column_batch(packed, cfg, members=members, ny=ny, nx=nx,
                                          dt=12, available_bytes=1 << 28)
    for step in range(2):
        result = bound()
        for member, state in enumerate(states):
            ordinary = microphysics.apply(state, cfg, 12)
            for name in names:
                value = getattr(state, name)
                expected = value.get()
                got = arrays[name][:, member * ny:(member + 1) * ny].get()
                assert got.tobytes() == expected.tobytes(), (step, member, name, terrain)
            for name in ("rainnc", "rainncv", "sr", "snownc", "snowncv", "graupelnc", "graupelncv"):
                got = getattr(result, name)[member * ny:(member + 1) * ny].get()
                expected = getattr(ordinary, name).get()
                assert got.tobytes() == expected.tobytes(), (step, member, name, terrain)


@pytest.mark.parametrize("members", [1, 4, 10, 20])
@pytest.mark.parametrize("specified,open_y", [(True, False), (False, False), (False, True)])
def test_member_aware_column_and_face_coupling_matches_each_stock_member(members, specified, open_y):
    import cupy as cp
    from types import SimpleNamespace
    from woof.core.physics import couple_ysu_tendencies, couple_column_tendencies
    from woof.ensemble.batch_physics import PreparedPhysicsCoupling
    rng = np.random.default_rng(9161)
    nz, ny, nx = 7, 6, 9
    c1h = cp.asarray(rng.uniform(0.3, 1, nz).astype(np.float32))
    c2h = cp.asarray(rng.uniform(1, 30, nz).astype(np.float32))
    metrics = {name: cp.asarray(rng.uniform(0.9, 1.2, shape).astype(np.float32))
               for name, shape in (("msft", (ny, nx)), ("msfu", (ny, nx + 1)), ("msfv", (ny + 1, nx)))}
    mut = cp.asarray(rng.uniform(80000, 100000, (members, ny, nx)).astype(np.float32))
    rates = {name: cp.asarray(rng.normal(0, 1e-4, (nz, members * ny, nx)).astype(np.float32))
             for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi")}
    cfg = SimpleNamespace(specified=specified, nested=False, open_x=False, open_y=open_y)
    bound = PreparedPhysicsCoupling(members=members, nz=nz, ny=ny, nx=nx,
        c1h=c1h, c2h=c2h, mut=mut.reshape(members * ny, nx), has_msf=True,
        available_bytes=1 << 24, **metrics)
    before = {name: array.get().tobytes() for name, array in rates.items()}
    got = bound.couple_ysu_tendencies(None, cfg, rates)
    for member in range(members):
        state = SimpleNamespace(p=cp.zeros((nz, ny, nx), cp.float32), c1h=c1h, c2h=c2h,
                                has_msf=True, total_mu=lambda m=member: mut[m], **metrics)
        ordinary = couple_ysu_tendencies(state, cfg, {name: array[:, member * ny:(member + 1) * ny]
                                                     .copy() for name, array in rates.items()})
        for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqi"):
            rows = ny + 1 if name == "rv" else ny
            actual = getattr(got, name)[:, member * rows:(member + 1) * rows].get()
            assert actual.tobytes() == getattr(ordinary, name).get().tobytes(), (member, name, specified, open_y)
    radiation = bound.couple_column_tendencies(None, cfg, rtheta=rates["dtheta"])
    for member in range(members):
        state = SimpleNamespace(p=cp.zeros((nz, ny, nx), cp.float32), c1h=c1h, c2h=c2h,
                                has_msf=True, total_mu=lambda m=member: mut[m], **metrics)
        ordinary = couple_column_tendencies(state, cfg, rtheta=rates["dtheta"][:, member * ny:(member + 1) * ny].copy())
        for name in ("ru", "rv", "rtheta", "rqv", "rqc"):
            rows = ny + 1 if name == "rv" else ny
            actual = getattr(radiation, name)[:, member * rows:(member + 1) * rows].get()
            assert actual.tobytes() == getattr(ordinary, name).get().tobytes(), (member, name)
    assert {name: array.get().tobytes() for name, array in rates.items()} == before


@pytest.mark.parametrize("members", [1, 4, 10, 20])
@pytest.mark.parametrize("terrain", [False, True])
def test_packed_physics_atmosphere_preserves_stock_words_without_member_seams(members, terrain):
    import cupy as cp
    from types import SimpleNamespace
    from test_mp_spec_zone_ring import _square_moist_state
    from woof.core.physics import _prepare_atmosphere
    from woof.ensemble.batch_physics import PreparedPackedPhysicsAtmosphere
    ny, nx, nz = 8, 9, 12
    states = []
    for member in range(members):
        state, cfg = _square_moist_state(ny=ny, nx=nx, nz=nz, mp=8, dt=12)
        state.thp += np.float32(member * 0.03125)
        state.u += np.float32(member * 0.0625)
        state.v += np.float32(member * 0.125)
        if terrain:
            patch = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) * np.float32(0.5)
            state.phb = cp.asarray(cp.asnumpy(state.phb)[:, None, None] + patch[None] * np.float32(9.81))
            state.thb = cp.asarray(cp.asnumpy(state.thb)[:, None, None] + patch[None] * np.float32(0.001))
        states.append(state)
    names = ("p", "thp", "alt", "u", "v", "php", "qv", "qc", "qr", "qi", "qs", "qg", "mup")
    fields = tuple(ColumnField(name, getattr(states[0], name).shape) for name in names)
    columns = ColumnBuffers(fields, members=members, available_bytes=1 << 28)
    packed = {name: columns.pack(name, cp.stack([getattr(state, name) for state in states])) for name in names}
    packed.update(thb=states[0].thb, phb=states[0].phb, qh=None)
    before = {name: value.get().tobytes() for name, value in packed.items() if value is not None}
    state = SimpleNamespace(**packed)
    mub2d = states[0].mub2d
    if mub2d is None:
        mub2d = cp.full((ny, nx), states[0].mub, cp.float32)
    bound = PreparedPackedPhysicsAtmosphere(state, members=members, ny=ny, nx=nx,
        c1h=states[0].c1h, c2h=states[0].c2h, dnw=states[0].dnw,
        p_top=states[0].p_top, mup=packed["mup"], mub2d=mub2d, available_bytes=1 << 28)
    bound()
    cp.cuda.get_current_stream().synchronize()
    pool_bytes = cp.get_default_memory_pool().used_bytes()
    got = bound()
    cp.cuda.get_current_stream().synchronize()
    assert cp.get_default_memory_pool().used_bytes() == pool_bytes
    for member, ordinary_state in enumerate(states):
        expected = _prepare_atmosphere(ordinary_state)
        for name, reference in expected.items():
            actual = got[name][:, member * ny:(member + 1) * ny].get()
            assert actual.tobytes() == reference.get().tobytes(), (member, name, terrain)
        assert bound.packed_mut[member * ny:(member + 1) * ny].get().tobytes() == ordinary_state.total_mu().get().tobytes()
    assert {name: value.get().tobytes() for name, value in packed.items() if value is not None} == before
