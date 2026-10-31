"""Short real-device proofs of the domain wrappers and continuation contract."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json

import cupy as cp
import numpy as np
import pytest

from conftest import requires_gpu
from woof.domain import Domain, WarmBubble, estimate_device_bytes

pytestmark = [pytest.mark.gpu, requires_gpu]


def build(**kwargs):
    args = dict(z=[0, 1000, 5000, 15000], theta=[300, 304, 325, 370],
                qv=[0.012, 0.010, 0.002, 0], u=[2, 3, 6, 6],
                nx=16, ny=16, nz=10, dx=1000, dy=1000, ztop=15000,
                dt=1.5, microphysics=1, bubble=WarmBubble())
    args.update(kwargs)
    return Domain.from_sounding(**args)


def differing_words(a, b):
    assert a.keys() == b.keys()
    return {name: int(np.count_nonzero(a[name].view(np.uint32) !=
                                      b[name].view(np.uint32))) for name in a}


def test_direct_step_and_snapshot_continuation(capsys):
    from woof.core.dycore import step
    direct, wrapped = build(), build()
    for _ in range(3):
        step(direct.state, direct.cfg)
    wrapped.step(3)
    names = wrapped.prognostic_names()
    words = differing_words(direct.to_host(names), wrapped.to_host(names))
    assert not any(words.values()), words
    checkpoint = wrapped.snapshot()
    wrapped.step(3)
    expected = wrapped.to_host(names)
    # Restore into a separately prepared object and into the advanced object.
    fresh = build()
    fresh.restore(checkpoint)
    fresh.step(3)
    restored_words = differing_words(expected, fresh.to_host(names))
    assert not any(restored_words.values()), restored_words
    wrapped.dt = 0.75
    wrapped.restore(checkpoint)
    assert wrapped.dt == 1.5 and wrapped.time == 4.5
    wrapped.step(3)
    assert not any(differing_words(expected, wrapped.to_host(names)).values())
    print(json.dumps({"direct_differing_words": words,
                      "restored_differing_words": restored_words,
                      "arrays": len(names), "steps_before": 3, "steps_after": 3}))


def test_placement_leaves_every_carried_word_unchanged():
    plain = build()
    epoch = datetime(2020, 6, 1, tzinfo=timezone.utc)
    placed = build(center_lat=35.5, center_lon=-97.0, start_time=epoch)
    assert plain.grid is None and plain.valid_time is None
    assert placed.grid is not None and placed.cfg == plain.cfg
    names = plain.prognostic_names()
    assert not any(differing_words(plain.to_host(names), placed.to_host(names)).values())
    plain.step(3)
    placed.step(3)
    words = differing_words(plain.to_host(names), placed.to_host(names))
    assert not any(words.values()), words
    assert placed.time == 4.5 and placed.valid_time.timestamp() == epoch.timestamp() + 4.5
    assert not placed.state.has_msf and not placed.state.rotational
    print(json.dumps({"placement_differing_words": words, "arrays": len(names), "steps": 3}))


@pytest.mark.parametrize("offset", [(3, -2), (-4, 2), (0, 0), (100, 100)])
def test_overlap_all_staggerings_and_exposed_cells(offset):
    src, dst = build(), build(nx=20, ny=18)
    names = ("u", "v", "w", "thp", "php", "mup", *src.moist_species())
    expected = {}
    for name in names:
        source = src.get(name)
        pattern = np.arange(source.size, dtype=np.float32).reshape(source.shape)
        src.set(name, pattern)
        dst.set(name, cp.full(dst.get(name).shape, -7, dtype=cp.float32))
        target = dst.to_host([name])[name]
        di, dj = offset
        for j in range(source.shape[-2]):
            for i in range(source.shape[-1]):
                if 0 <= j + dj < target.shape[-2] and 0 <= i + di < target.shape[-1]:
                    target[..., j + dj, i + di] = pattern[..., j, i]
        expected[name] = target
    field = next(n for n, a in src.state.physics.fields.items()
                 if a.shape == (src.cfg.ny, src.cfg.nx))
    src.state.physics.fields[field][...] = 12
    dst.state.physics.fields[field][...] = -7
    dst.copy_overlap(src, *offset)
    for name, wanted in expected.items():
        np.testing.assert_array_equal(dst.to_host([name])[name], wanted, err_msg=name)
    wanted = np.full((dst.cfg.ny, dst.cfg.nx), -7, dtype=np.float32)
    di, dj = offset
    for j in range(src.cfg.ny):
        for i in range(src.cfg.nx):
            if 0 <= j + dj < dst.cfg.ny and 0 <= i + di < dst.cfg.nx:
                wanted[j + dj, i + di] = 12
    np.testing.assert_array_equal(cp.asnumpy(dst.state.physics.fields[field]), wanted)


def test_fields_time_boundaries_and_radiation():
    d = build(boundary="specified", overrides={"ra_physics": 90},
              physics_options={"radiation_start_time": datetime(2020, 6, 1, 12, tzinfo=timezone.utc),
                               "radiation_latitude": np.full((16, 16), 40.0),
                               "radiation_longitude": np.full((16, 16), -100.0)})
    assert d.state._scratch is not None
    assert hasattr(d.state.physics, "radiation_due_override")
    assert {"qv", "qc", "qr"} <= set(d.moist_species())
    with pytest.raises(ValueError, match="shape"):
        d.set("w", np.zeros((1,)))
    with pytest.raises(KeyError):
        d.get("_scratch")
    d.diagnostics()
    assert d.set("u", d.get("u")) is d.get("u")
    before = d.boundary_snapshot()
    sides = {s: d.extract_side(s) for s in ("west", "east", "south", "north")}
    d.set_lateral_boundaries(first=sides, second=sides, start_seconds=0, end_seconds=60)
    np.testing.assert_array_equal(sides["east"]["u"], before["u"][..., -5:][..., ::-1])
    d.state.physics.radiation_due_override = False
    previous_calls = d.state.physics.call_counts["radiation"]
    d.force_radiation_next_step()
    pending = d.snapshot()
    d.step()
    assert d.state.physics.call_counts["radiation"] == previous_calls + 1
    assert d.state.physics.radiation_due_override is False
    d.restore(pending)
    d.step()
    assert d.state.physics.call_counts["radiation"] == previous_calls + 1
    assert d.state.physics.radiation_due_override is False
    d.step()
    assert d.state.physics.call_counts["radiation"] == previous_calls + 1
    old_dt = d.dt
    d.advance(3.5)
    assert d.time == 6.5 and d.dt == old_dt
    assert estimate_device_bytes(16, 16, 10, d.cfg) > 0
    assert d.carried_arrays()["state"]["u"] == "serialize"
    wrapped = Domain.from_state(d.state, d.cfg)
    assert wrapped.time == d.time
    with pytest.raises(ValueError):
        Domain.from_state(d.state, replace(d.cfg, nx=17))
