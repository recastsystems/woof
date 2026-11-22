"""What a new user types first: a swapped ``--point`` and a cycle days ahead.

THE BREAKAGES THESE PREVENT (gpuwm 3.0.3 candidate check, 2026-11-15, the same
doors in WOOF-ArWen): ``--point -90,37.5`` (meant 37.5 N, 90 W) built a domain on the
South Pole without a word, and a cycle typed five days ahead was answered "usually
posts around 03:30Z, and it is 16:00Z", which reads as later today.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import country_door, init_sources, mesh_point  # noqa: E402
from woof.hex.errors import MpasPortError  # noqa: E402
from woof.hex.mesh_point import PointPlanRefusal  # noqa: E402


def test_a_point_at_a_pole_with_a_longitude_is_refused_naming_the_swap() -> None:
    with pytest.raises(PointPlanRefusal, match=r"South Pole.*did you mean --point 37.5,-90\?"):
        mesh_point.parse_point("-90,37.5")
    with pytest.raises(PointPlanRefusal, match=r"North Pole.*did you mean --point 45,89.7\?"):
        mesh_point.parse_point("89.7,45")


def test_the_pole_itself_and_ordinary_points_still_parse() -> None:
    assert mesh_point.parse_point("-90,0") == (-90.0, 0.0)
    assert mesh_point.parse_point("37.5,-90") == (37.5, -90.0)
    assert mesh_point.parse_point("-89,37.5") == (-89.0, 37.5)
    assert mesh_point.parse_point("39.1, 265.42") == (39.1, pytest.approx(-94.58))


def test_a_latitude_out_of_range_that_reads_as_a_longitude_says_so() -> None:
    with pytest.raises(PointPlanRefusal, match=r"did you mean --point 35.2,-97.5\?"):
        mesh_point.parse_point("-97.5,35.2")
    with pytest.raises(PointPlanRefusal) as refused:
        mesh_point.parse_point("95,200")
    assert "did you mean" not in str(refused.value)


def test_a_southern_or_western_point_after_a_space_reaches_its_flag() -> None:
    from woof.hex.cli import build_parser, join_negative_coordinates

    argv = join_negative_coordinates(["regional", "--point", "-33.9,151.2", "--ellipse", "-33.9,151.2,300,200",
                                      "--cycle", "2026-11-15T12", "--work-dir", "w"])
    assert argv[1:3] ==["--point=-33.9,151.2", "--ellipse=-33.9,151.2,300,200"]
    arguments = build_parser("regional").parse_args(argv)
    assert (arguments.point, arguments.ellipse) == ("-33.9,151.2", "-33.9,151.2,300,200")
    assert join_negative_coordinates(["regional", "--point", "--help"]) == ["regional", "--point", "--help"]
    assert join_negative_coordinates(["x", "--point", "-a,b"]) == ["x", "--point", "-a,b"]


def _country_args(point: str) -> SimpleNamespace:
    return SimpleNamespace(point=point, radius_km=500.0, region=None, polygon=None,
                           shape=None, ellipse=None, table=country_door.TABLE, row=None)


def test_the_country_door_point_cap_refuses_the_same_swap() -> None:
    with pytest.raises(MpasPortError, match=r"South Pole.*did you mean --point 37.5,-90\?"):
        country_door.region_request(_country_args("-90,37.5"))
    with pytest.raises(MpasPortError, match=r"read the other way round 35.2,-97.5 is a valid point"):
        country_door.region_request(_country_args("-97.5,35.2"))


def _ladder_refusal(due: datetime, now: datetime) -> str:
    plan = SimpleNamespace(url_ready=[None], due_utc=due.isoformat(), lag_minutes=210.0,
                           label="gfs 2026-11-20T00Z f000", filename="gfs.t00z.pgrb2.0p25.f000")

    def fetch(data_url, index_url):
        raise init_sources.InitSourceRefusal(f"{data_url} answered HTTP 404")

    with pytest.raises(init_sources.InitSourceRefusal) as refused:
        init_sources._ladder(plan, [("https://noaa-gfs-bdp-pds.s3.amazonaws.com/x", None)], fetch, now=now)
    return str(refused.value)


def test_a_cycle_days_ahead_is_refused_with_its_date_and_the_wait() -> None:
    now = datetime(2026, 11, 15, 16, 0, tzinfo=timezone.utc)
    text = _ladder_refusal(datetime(2026, 11, 20, 3, 30, tzinfo=timezone.utc), now)
    assert "usually posts around 2026-11-20 03:30Z (posting lag 210 min), and it is 16:00Z: 4 d 12 h from now." in text


def test_a_cycle_later_today_keeps_the_clock_time() -> None:
    now = datetime(2026, 11, 15, 16, 0, tzinfo=timezone.utc)
    text = _ladder_refusal(now + timedelta(minutes=95), now)
    assert "usually posts around 17:35Z (posting lag 210 min), and it is 16:00Z: 1 h 35 min from now." in text
