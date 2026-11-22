"""A storm-following case names the source member it starts from (`moving case --member`, written as init_member).

THE BREAKAGE THIS PREVENTS (2026-11-13 walk of the hex orders the site accepts): a storm-following hex run from GEFS
had no way to name its member, so `moving stage` refused it after the order: "source 'gefs' is an ensemble; pass
--member".
"""
import json
from pathlib import Path

import pytest

from woof.hex.cli import main
from woof.hex.moving_author import author_case
from woof.hex.moving_case import MovingCaseError, load_case
from woof.hex.moving_members import stage_case

TEMPLATE = Path(__file__).parents[1] / "verification/moving48/cases/tc-wp-2025110812.json"


def point_case(tmp_path, *extra, source="gefs"):
    out = tmp_path / "case.json"
    assert main(["moving", "case", "--point", "37.0,-98.0", "--target", "low", "--check-minutes", "30",
                 "--cycle", "2026-11-13T00", "--out", str(out), "--source", source, "--hours", "6",
                 "--fine-km", "3", "--fine-radius-km", "150", "--background-km", "96", *extra]) == 0
    return out


def fake_stage(monkeypatch, tmp_path):
    from woof.hex import init_sources
    acquired = []

    def acquire(plans, *args, **kwargs):
        acquired.extend(plans)
        return [dict(file=str(tmp_path / p.filename)) for p in plans]

    met = tmp_path / "met"
    met.write_text("decoded fixture")
    monkeypatch.setattr(init_sources, "acquire", acquire)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda _: "unused")
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda _: "unused")
    monkeypatch.setattr(init_sources, "decode", lambda *args: dict(met=str(met)))
    return acquired


def test_console_member_is_the_case_init_member_and_stage_reads_the_control(tmp_path, monkeypatch):
    case = load_case(point_case(tmp_path, "--member", "0"))
    assert case["init_source"] == "gefs" and case["init_member"] == 0
    acquired = fake_stage(monkeypatch, tmp_path)
    stage = stage_case(tmp_path / "case.json", tmp_path / "prepared", log=lambda _: None)
    assert stage["source_member"] == 0
    assert acquired and all("gec00" in p.url for p in acquired)


def test_perturbed_member_reaches_the_stage(tmp_path, monkeypatch):
    point_case(tmp_path, "--member", "7")
    acquired = fake_stage(monkeypatch, tmp_path)
    assert stage_case(tmp_path / "case.json", tmp_path / "prepared", log=lambda _: None)["source_member"] == 7
    assert acquired and all("gep07" in p.url for p in acquired)


def test_without_member_a_gefs_case_is_still_refused_at_stage(tmp_path):
    from woof.hex.init_sources import InitSourceRefusal
    case = load_case(point_case(tmp_path))
    assert "init_member" not in case
    with pytest.raises(InitSourceRefusal, match="is an ensemble; pass --member"):
        stage_case(tmp_path / "case.json", tmp_path / "prepared", log=lambda _: None)


def test_member_out_of_the_declared_range_is_refused_before_any_fetch(tmp_path, monkeypatch):
    from woof.hex.init_sources import InitSourceRefusal
    point_case(tmp_path, "--member", "31")
    acquired = fake_stage(monkeypatch, tmp_path)
    with pytest.raises(InitSourceRefusal, match="outside.*declared members"):
        stage_case(tmp_path / "case.json", tmp_path / "prepared", log=lambda _: None)
    assert acquired == []


def test_member_on_a_source_without_members_is_refused_before_writing(tmp_path):
    out = tmp_path / "gfs.json"
    with pytest.raises(MovingCaseError, match="declares no members"):
        author_case(out, point=(37.0, -98.0), target="low", cycle="2026-11-13T00", source="gfs", hours=6,
                    member=0)
    assert not out.exists()


@pytest.mark.parametrize("member", [-1, True, 0.0, "0"])
def test_member_must_be_a_nonnegative_integer(tmp_path, member):
    out = tmp_path / "bad.json"
    with pytest.raises(MovingCaseError, match="nonnegative integer"):
        author_case(out, point=(37.0, -98.0), target="low", cycle="2026-11-13T00", source="gefs", hours=6,
                    member=member)
    assert not out.exists()


def test_a_template_member_does_not_carry_into_a_new_case(tmp_path):
    template = tmp_path / "template.json"
    body = json.loads(TEMPLATE.read_text())
    body.update(init_source="gefs", init_member=5)
    template.write_text(json.dumps(body))
    case = author_case(tmp_path / "plain.json", point=(37.0, -98.0), target="low", cycle="2026-11-13T00",
                       source="gfs", hours=6, template=template)
    assert "init_member" not in case
    case = author_case(tmp_path / "control.json", point=(37.0, -98.0), target="low", cycle="2026-11-13T00",
                       source="gefs", hours=6, template=template, member=0)
    assert case["init_member"] == 0


def test_gfs_case_without_member_is_unchanged(tmp_path):
    case = load_case(point_case(tmp_path, source="gfs"))
    assert case["init_source"] == "gfs" and "init_member" not in case
