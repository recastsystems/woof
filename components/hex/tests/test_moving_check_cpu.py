"""Replay the optimized operators on the actual moving-check initial state."""
import ast
from pathlib import Path

import numpy as np
import pytest

from woof.hex import driver, dynamics_v841, engines, moving_jolt_check as check


def test_cpu_operators_match_scalar_reference_on_moving_check(tmp_path, monkeypatch):
    for spec in (engines.MESH, engines.GEOMETRY):
        if engines.locate(spec)[0] is None:
            pytest.skip(f"{spec.name} is needed to generate the actual moving-check fixture")
    family = check.build_family(tmp_path)
    mesh, cfg, coeff, winds = check.setup(family["meshes"][0])
    state, saved, prep = check.fixture(mesh, check.terrain_for(mesh), amp=1e-5)
    integ = check._Integrator(device="cpu", lts=False, cfg=cfg, coeff=coeff, winds=winds, rate_shift=0)
    live = integ.driver(state, saved, prep)
    source = ast.parse((Path(__file__).parent / "data/moving_cpu_scalar_reference.py").read_text())
    calls = {}
    for node in source.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        module = dynamics_v841 if node.name.startswith("vector_") else driver
        namespace = dict(module.__dict__)
        # Annotations are not needed to replay the frozen arithmetic.
        node.returns = None
        for arg in (*node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
        exec(compile(ast.Module(body=[node], type_ignores=[]), "scalar-reference", "exec"), namespace)
        scalar, vector = namespace[node.name], getattr(driver, node.name)
        def compare(*args, _name=node.name, _scalar=scalar, _vector=vector, **kwargs):
            expected, actual = _scalar(*args, **kwargs), _vector(*args, **kwargs)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes(), _name
            if _name.startswith("vector_") and _name not in calls:
                # The actual fixture has resting reference winds. Exercise
                # nonzero winds too, where scalar/SIMD trig rounding matters.
                for dtype in (np.float32, np.float64):
                    varied = {k: v.astype(dtype) if isinstance(v, np.ndarray) else v
                              for k, v in kwargs.items()}
                    nlev = varied["normal_velocity"].shape[0]
                    varied["reference_wind"] = dynamics_v841.V841ReferenceWindProfiles(
                        np.linspace(-7, 11, nlev, dtype=dtype), np.linspace(13, -3, nlev, dtype=dtype))
                    assert _scalar(*args, **varied).tobytes() == _vector(*args, **varied).tobytes()
            calls[_name] = calls.get(_name, 0) + 1
            return actual
        monkeypatch.setattr(driver, node.name, compare)
    result, _ = integ.advance(live, state, saved)
    assert all(count == 9 for count in calls.values()) and len(calls) == 3
    assert np.isfinite(result.rho).all()


def test_cpu_arm_workers_respect_available_budget(monkeypatch):
    monkeypatch.setattr("woof.hex.cpu_budget.available_cpus", lambda: 24)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("RAYON_NUM_THREADS", "8")
    assert check._cpu_workers(check.ARMS) == 3
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert check._cpu_workers(check.ARMS) == 2
