"""The moving start member is rotated from the premade family template, never generated on the start path.

THE BREAKAGE THIS PREVENTS: every storm- and setup-following run generated its start mesh on the box before the
first step (premade meshes; production run_312c8803de02ae46 spent 940 s preparing the start
member on a new box). The template is keyed by its generator spec alone, so the image's prep store carries it.
"""
import json
from pathlib import Path

import pytest

from woof.hex import moving_members as m, prep_store
from woof.hex.moving_author import mesh_block


def _case(tmp_path, center=(11.66, -102.51), corridor=False):
    mesh = mesh_block(3.0, 96.0, 200.0)
    if corridor:
        mesh["corridor"] = {"cross_track_km": 450.0}
    case = {"start_time": "2026-10-24T00:00:00", "dt_seconds": 18.75, "n_levels": 55,
            "vertical_spec": "tc55-v1.json", "storm": {"initial_center_deg": list(center)}, "mesh": mesh}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return path, case


def test_template_spec_is_the_case_family_at_the_template_centre(tmp_path):
    _, case = _case(tmp_path)
    spec = m.template_spec(case)
    assert spec["name"] == m.TEMPLATE_NAME
    assert all(r["shape"]["center_deg"] == list(m.TEMPLATE_CENTER) for r in spec["regions"])
    # the same rings as the member a storm centre gets, only the centre differs
    member = m.member_spec(case, m.TEMPLATE_NAME, 11.66, -102.51)
    strip = lambda s: [{**r, "shape": {**r["shape"], "center_deg": None}} for r in s["regions"]]
    assert strip(spec) == strip(member)


@pytest.mark.parametrize("corridor, center, rotated", [
    (False, (11.66, -102.51), True),     # a storm: rotated from the template
    (True, (11.66, -102.51), False),     # a corridor family is stitched, not rotated
    (False, (0.5, 179.5), False),        # within 2 degrees of the template's antipode: generated
])
def test_prepare_case_rotates_the_start_member(tmp_path, monkeypatch, corridor, center, rotated):
    path, case = _case(tmp_path, center, corridor)
    monkeypatch.setattr(m, "load_case", lambda _: case)
    monkeypatch.setattr(m, "read_stage", lambda _: {})
    monkeypatch.setattr(m, "frame_spec", lambda *a, **k: {})
    monkeypatch.setattr(m, "generate_mesh", lambda *a, **k: 0.0)
    calls = {}
    monkeypatch.setattr(m, "rotation_template", lambda case, folder, **k: calls.setdefault("template", folder))

    def member(case_path, folder, name, lat, lon, **kw):
        calls["member"] = kw.get("rotate_from")
        return {"name": name}

    monkeypatch.setattr(m, "prepare_member", member)
    for name in ("frame", "template"):
        (tmp_path / f"{name}-mesh.json").write_text("{}")
        (tmp_path / f"{name}.grid.nc").write_bytes(b"g")
    m.prepare_case(path, tmp_path, log=lambda _: None)
    assert ("template" in calls) == rotated
    assert calls["member"] == (m.TEMPLATE_NAME if rotated else None)


def test_rotation_template_fetches_from_the_store_and_builds_only_on_a_miss(tmp_path, monkeypatch):
    _, case = _case(tmp_path)
    monkeypatch.setattr(m, "_tool", lambda key: "rw_mpas_mesh")
    monkeypatch.setattr(m, "_stamp", lambda *a: None)
    seen = []

    def run_or_fetch(argv, kind=None, runner=None):
        seen.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.write_bytes(b"grid")
        Path(argv[argv.index("--receipt") + 1]).write_text("{}")
        return {"source": "store"}

    monkeypatch.setattr(prep_store, "run_or_fetch", run_or_fetch)
    grid = m.rotation_template(case, tmp_path, log=lambda _: None)
    assert grid.read_bytes() == b"grid" and len(seen) == 1
    assert json.loads((tmp_path / f"{m.TEMPLATE_NAME}.spec.json").read_text()) == m.template_spec(case)
    m.rotation_template(case, tmp_path, log=lambda _: None)        # present: not asked again
    assert len(seen) == 1
    other = dict(case, mesh=mesh_block(3.0, 96.0, 250.0))          # another radius: a new template
    m.rotation_template(other, tmp_path, log=lambda _: None)
    assert len(seen) == 2


def test_near_antipode():
    assert m._near_antipode((0.0, 179.0), (0.0, 0.0))
    assert not m._near_antipode((25.0, -80.0), (0.0, 0.0))
    assert not m._near_antipode((0.0, 170.0), (0.0, 0.0))
