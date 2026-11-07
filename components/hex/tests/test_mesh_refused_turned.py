"""A cap family the generator refuses at its point is built about another centre and turned to the point."""
import json
import subprocess
from pathlib import Path

import pytest

from woof.hex import moving_mesh, prep_store
from woof.hex.errors import MpasPortError
from woof.hex.mesh_point import ladder_spec


def _argv(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return ["/x/rw_mpas_mesh", "--spec", str(path), "--out", str(tmp_path / "g.nc"), "--receipt", str(tmp_path / "r.json"), "--clobber"]


def _builder(refused_centres, seen):
    def run(argv):
        spec = json.loads(Path(argv[argv.index("--spec") + 1]).read_text())
        centre = tuple(spec["regions"][0]["shape"]["center_deg"]) if spec["regions"] else None
        seen.append(centre)
        if centre in refused_centres:
            raise subprocess.CalledProcessError(1, argv)
        Path(argv[argv.index("--out") + 1]).write_bytes(b"grid about " + repr(centre).encode())
        Path(argv[argv.index("--receipt") + 1]).write_text("{}")
        return subprocess.CompletedProcess(argv, 0)
    return run


@pytest.fixture(autouse=True)
def _no_store(monkeypatch):
    monkeypatch.setattr(prep_store, "configured_stores", lambda: [])
    monkeypatch.setattr(prep_store, "outbox", lambda: None)
    turned = []

    def rotate(source, output, *, start, end, file_id=None):
        Path(output).write_bytes(Path(source).read_bytes() + f" turned {start}->{end}".encode())
        turned.append((tuple(start), tuple(end)))
        return {"sha256": "x", "wall_seconds": 0.0}
    monkeypatch.setattr(moving_mesh, "rotate_grid", rotate)
    return turned


SPEC = ladder_spec((-12.46, 130.84), fine_dx_m=937.5, radius_km=60.0, background_km=120.0, name="n")


def test_a_mesh_that_builds_at_its_point_is_untouched(tmp_path, _no_store):
    seen = []
    prep_store.run_or_fetch(_argv(tmp_path, SPEC), runner=_builder(set(), seen))
    assert seen == [(-12.46, 130.84)] and not _no_store
    assert (tmp_path / "g.nc").read_bytes() == b"grid about (-12.46, 130.84)"


def test_a_refused_mesh_comes_from_the_first_centre_that_settles_turned_to_the_point(tmp_path, _no_store):
    seen = []
    prep_store.run_or_fetch(_argv(tmp_path, SPEC), runner=_builder({(-12.46, 130.84), (0.0, 0.0)}, seen))
    assert seen == [(-12.46, 130.84), (0.0, 0.0), (10.0, 20.0)]
    assert _no_store == [((10.0, 20.0), (-12.46, 130.84))]
    assert (tmp_path / "g.nc").read_bytes().endswith(b"(-12.46, 130.84)")
    assert json.loads((tmp_path / "r.json").read_text())["built_about"] == [10.0, 20.0]
    assert not list(tmp_path.glob("*.about-*.grid.nc"))


def test_the_refusal_stands_when_no_centre_builds_it(tmp_path):
    every = {(-12.46, 130.84), *prep_store.REBUILD_CENTRES}
    with pytest.raises(subprocess.CalledProcessError):
        prep_store.run_or_fetch(_argv(tmp_path, SPEC), runner=_builder(every, []))


def test_a_spec_that_is_no_cap_family_is_not_retried(tmp_path, _no_store):
    seen = []
    with pytest.raises(subprocess.CalledProcessError):
        drawn = {"background_km": 120, "regions": [{"shape": {"kind": "polygon", "vertices_deg": [[0, 0], [1, 0], [1, 1]]}, "spacing_km": 3, "transition_km": 54}]}
        def run(argv):
            seen.append(1)
            raise subprocess.CalledProcessError(1, argv)
        prep_store.run_or_fetch(_argv(tmp_path, drawn), runner=run)
    assert seen == [1] and not _no_store


def test_a_door_whose_runner_returns_the_exit_code_or_raises_its_own_error_gets_the_same(tmp_path, _no_store):
    def by_code(argv):
        spec = json.loads(Path(argv[argv.index("--spec") + 1]).read_text())
        if spec["regions"][0]["shape"]["center_deg"] == [-12.46, 130.84]:
            return subprocess.CompletedProcess(argv, 1)
        Path(argv[argv.index("--out") + 1]).write_bytes(b"g")
        return subprocess.CompletedProcess(argv, 0)
    assert prep_store.run_or_fetch(_argv(tmp_path, SPEC), runner=by_code)["completed"].returncode == 0

    def by_error(argv):
        spec = json.loads(Path(argv[argv.index("--spec") + 1]).read_text())
        if spec["regions"][0]["shape"]["center_deg"] == [-12.46, 130.84]:
            raise MpasPortError("mesh exited 1")
        Path(argv[argv.index("--out") + 1]).write_bytes(b"g")
        return subprocess.CompletedProcess(argv, 0)
    (tmp_path / "g.nc").unlink()
    assert prep_store.run_or_fetch(_argv(tmp_path, SPEC), runner=by_error)["completed"].returncode == 0
