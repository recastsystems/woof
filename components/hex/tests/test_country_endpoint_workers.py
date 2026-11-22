"""A hex region through the country door reads as many source hours at once as the host has room for.

THE BREAKAGE THESE PREVENT (WOOF hex regions, 2026-11-13): the country door read two source hours (the start and each
boundary time) at a time on every host. An ERA5 region downloads a whole-globe analysis for each boundary hour and an
ICON region about 120 DWD objects, so on a 16-vCPU box most of the prepare stage was two downloads at a time with the
line and the CPUs mostly idle. The default is now the regional door's rule for its leads: half the CPUs, at least 2,
at most 8. Every test below but the explicit-value one fails on woof hex 0.5.0 (default 2 on every host).
"""
import pytest

from woof.hex.cli import build_parser
from woof.hex.errors import MpasPortError

ERA5_REGION = ["country", "--point", "39.080,-98.040", "--radius-km", "2650", "--source", "era5",
               "--cycle", "2011-04-26T06", "--hours", "6", "--history-every-minutes", "60", "--skip", "score",
               "--forecast-arg=--physics-suite=convection_permitting"]


def test_the_rule_is_half_the_cpus_at_least_two_at_most_eight():
    from woof.hex.country_door import country_endpoint_workers
    assert country_endpoint_workers(cpus=16) == 8   # a 16-core box
    assert country_endpoint_workers(cpus=8) == 4    # an 8-core box
    assert country_endpoint_workers(cpus=24) == 8
    assert country_endpoint_workers(cpus=4) == 2
    assert country_endpoint_workers(cpus=1) == 2


def test_an_explicit_value_is_kept_and_a_non_positive_one_refused():
    from woof.hex.country_door import country_endpoint_workers
    assert country_endpoint_workers(3, cpus=16) == 3
    assert country_endpoint_workers(2, cpus=16) == 2
    with pytest.raises(MpasPortError, match="endpoint workers must be positive"):
        country_endpoint_workers(0, cpus=16)


def test_the_parser_leaves_the_choice_to_the_door():
    args = build_parser().parse_args(ERA5_REGION + ["--work-dir", "unused"])
    assert args.endpoint_workers is None


@pytest.mark.parametrize("cpus,expected", [(16, 8), (8, 4)])
def test_a_plain_country_region_reads_its_hours_by_the_host(tmp_path, monkeypatch, cpus, expected):
    from woof.hex import cpu_budget
    from woof.hex.country_door import plan
    monkeypatch.setattr(cpu_budget, "available_cpus", lambda: cpus)
    args = build_parser().parse_args(ERA5_REGION + ["--work-dir", str(tmp_path / "run")])
    value = plan(args)
    assert args.endpoint_workers == expected
    assert len(value["boundary_endpoints"]) == 7  # the start and six hourly ERA5 analyses


def test_the_site_argv_with_a_value_keeps_it(tmp_path, monkeypatch):
    from woof.hex import cpu_budget
    from woof.hex.country_door import plan
    monkeypatch.setattr(cpu_budget, "available_cpus", lambda: 16)
    args = build_parser().parse_args(ERA5_REGION + ["--work-dir", str(tmp_path / "run"), "--endpoint-workers", "2"])
    plan(args)
    assert args.endpoint_workers == 2
