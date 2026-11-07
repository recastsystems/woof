"""Requested vertical resolution and streaming use the actual emitted route."""
from datetime import datetime
from dataclasses import replace
import argparse
import json
import tomllib

import numpy as np
import pytest

from woof import domain_wizard as wizard
from woof.core import streaming
from woof.core.grid import make_vertical_coord, resample_eta_levels
from tilestream.autoplan import Machine


START = datetime(2026, 7, 29, 18)
FREE = int(11.25 * wizard.GIB)


@pytest.fixture
def declared_machine(monkeypatch):
    # Declare only the available resources. The actual planner and every
    # physics/ingest estimate run unchanged; device detection must not run.
    calls = []
    def machine(**kwargs):
        calls.append(kwargs)
        return Machine(vram_bytes=kwargs["vram_bytes"],
                       host_bytes=128 * wizard.GIB, name=kwargs["name"])
    monkeypatch.setattr(streaming, "planner_machine", machine)
    def detect(*args, **kwargs):
        pytest.fail("a declared wizard card must never probe the local GPU")
    monkeypatch.setattr(Machine, "detect", detect)
    return calls


def config(*, nz=None, tiles=None, dims=None, ratios=()):
    return wizard.render_config(
        name="resolution", start_time=START, hours=6,
        projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=dims or [(400, 400)], ratios=ratios,
        fetch_hints=wizard._candidate_fetch_hints("gfs"), case_data=None,
        root_dx_m=3000.0, nz=nz, tiles=tiles)


def test_default_ladder_is_byte_identical_and_explicit_49_is_identical():
    text = config()
    assert config(nz=49) == text
    raw = tomllib.loads(text)
    assert raw["shared"]["nz"] == 49
    assert raw["shared"]["eta_levels"] == list(wizard._ETA_LEVELS)
    assert "tiles" not in raw


def test_resampling_preserves_stretching_and_input():
    original = np.array([1., .9, .5, 0.])
    result = resample_eta_levels(original, 6)
    np.testing.assert_allclose(result, [1., .95, .9, .7, .5, .25, 0.], atol=1e-15)
    np.testing.assert_array_equal(original, [1., .9, .5, 0.])
    result[0] = -1
    assert original[0] == 1


@pytest.mark.parametrize("nz", [0, -1, 2.5, True, float("nan")])
def test_shared_resampler_rejects_invalid_counts(nz):
    with pytest.raises(ValueError, match="positive integer"):
        resample_eta_levels([1., .5, 0.], nz)


@pytest.mark.parametrize("eta", [[1., .5, .5, 0.], [1., float("nan"), 0.],
                                  [.9, .5, 0.], [1., .5, .1], [], [[1., 0.]]])
def test_shared_resampler_rejects_invalid_source_ladder(eta):
    with pytest.raises(ValueError, match="eta_levels"):
        resample_eta_levels(eta, 76)


def test_76_levels_load_through_the_actual_model_coordinate():
    exp = wizard.experiment_from_text(config(nz=76), source="<76-level test>")
    run = exp.domains[0].run
    assert run.nz == 76
    coord = make_vertical_coord(run.nz, eta_levels=exp.vertical.eta_levels)
    assert coord.znw.shape == (77,)
    assert coord.znw[0] == 1. and coord.znw[-1] == 0.
    assert np.all(coord.dnw < 0.)
    assert abs(coord.dnw[0]) < abs(coord.dnw[38])


def test_cli_flags_defaults_and_bare_tiles():
    parser = argparse.ArgumentParser()
    wizard.register_cli(parser.add_subparsers())
    required = ["domain", "--point", "35.3,-97.5", "--cycle", "2026-07-29T18",
                "--out", "area.toml"]
    baseline = parser.parse_args(required)
    assert baseline.nz is None and baseline.tiles is None
    requested = parser.parse_args(required + ["--nz", "76", "--tiles"])
    assert requested.nz == 76 and requested.tiles == "auto"
    forced = parser.parse_args(required + ["--tiles", "on"])
    assert forced.tiles == "on"


def test_auto_sizing_preserves_resident_capacity_and_grows_with_declared_cards(declared_machine):
    sizes = []
    for capacity, free_gib in [(8., 7.25), (16., 15.04), (24., 22.56)]:
        kwargs = dict(ladder="12", free_bytes=int(free_gib * wizard.GIB),
                      vram_gib=capacity, hours=6, start_time=START,
                      projection=wizard._projection_entries(35.3, -97.5, "auto"),
                      source="gfs", name="auto-capacity")
        resident, _ = wizard.fit_ladder(**kwargs, tiles="off")
        automatic, exp = wizard.fit_ladder(**kwargs, tiles="auto")
        resident_cells = resident[0][0] * resident[0][1]
        automatic_cells = automatic[0][0] * automatic[0][1]
        assert automatic_cells >= resident_cells, (capacity, resident, automatic)
        phases = wizard._sizing_phases(exp, free_bytes=kwargs["free_bytes"],
                                       source="gfs", vram_gib=capacity)
        budget = wizard.sizing_budget_bytes(exp, free_bytes=kwargs["free_bytes"],
                                            vram_gib=capacity,
                                            forcing_interval_seconds=10800.)
        assert 0 < phases.peak_envelope_bytes <= budget
        sizes.append(automatic_cells)
    assert sizes == sorted(sizes)


def test_fixed_polygon_requires_streaming_at_the_same_card_budget(
        tmp_path, declared_machine):
    path = tmp_path / "area.geojson"
    path.write_text(json.dumps({"type": "Polygon", "coordinates": [[
        [-104., 31.], [-91., 31.], [-91., 39.], [-104., 39.], [-104., 31.]]]}))
    footprint = wizard.load_polygon_footprint(path)
    kwargs = dict(footprint=footprint, buffers_km=(0.,), free_bytes=FREE,
                  hours=6, start_time=START,
                  projection=wizard._projection_entries(35.3, -97.5, "auto"),
                  source="gfs", name="fixed", ratios=(), root_dx_m=3000.,
                  nz=76, vram_gib=12.)
    with pytest.raises(wizard.DomainFitError, match="polygon.*requires"):
        wizard.fit_polygon_ladder(**kwargs)
    dims, exp = wizard.fit_polygon_ladder(**kwargs, tiles="auto")
    assert exp.domains[0].run.nz == 76
    assert exp.tiles.mode == "auto"
    wizard.verify_polygon_containment(exp, footprint, (0.,))
    phases = wizard._sizing_phases(exp, free_bytes=FREE, source="gfs",
                                   vram_gib=12., forcing_interval_seconds=10800.)
    budget = wizard.sizing_budget_bytes(exp, free_bytes=FREE, vram_gib=12.,
                                          forcing_interval_seconds=10800.)
    assert phases.streamed is not None
    assert phases.peak_envelope_bytes <= budget
    assert phases.resident_forecast_envelope_bytes > budget
    assert phases.ingest is not None
    assert {call["vram_bytes"] for call in declared_machine} == {FREE}


def test_more_levels_are_priced_before_the_point_fit(declared_machine):
    kwargs = dict(ratios=(), free_bytes=FREE, hours=6, start_time=START,
                  projection=wizard._projection_entries(35.3, -97.5, "auto"),
                  source="gfs", name="point", root_dx_m=3000., vram_gib=12.)
    dims49, exp49 = wizard.fit_ladder(**kwargs)
    dims76, exp76 = wizard.fit_ladder(**kwargs, nz=76)
    assert exp49.domains[0].run.nz == 49
    assert exp76.domains[0].run.nz == 76
    assert dims76[0][0] * dims76[0][1] < dims49[0][0] * dims49[0][1]


def test_streaming_does_not_remove_the_ingest_admission_term(declared_machine):
    # The ingest term is the ITEMIZED preparation price (A65: model state,
    # forcing analysis, vertical setup, their temporaries, allocator
    # headroom and the context), not a multiple of the forecast state.
    # Itemized, a 600x600x76 GFS preparation prices at 10.75 GB, under this
    # 12 GiB card's 11.54 GB budget; 700x700x76 prices at 14.3 GB, over it,
    # so the term still binds here and the assertions below keep their
    # meaning.
    exp = wizard.experiment_from_text(
        config(nz=76, tiles="auto", dims=[(700, 700)]), source="<large>")
    budget = wizard.sizing_budget_bytes(exp, free_bytes=FREE, vram_gib=12.,
                                          forcing_interval_seconds=10800.)
    # Prepared on the card: the device ingest term stays and binds.
    phases = wizard._sizing_phases(exp, free_bytes=FREE, source="gfs", vram_gib=12.,
                                   preprocess_backend="cuda")
    assert phases.streamed is not None
    assert phases.forecast_envelope_bytes < phases.ingest_envelope_bytes
    assert phases.peak_envelope_bytes == phases.ingest_envelope_bytes > budget
    # Prepared on the host, which a [tiles] declaration selects for this
    # source: the card holds nothing in that phase, and the same working
    # set is host RAM, still priced and still an admission term.
    phases = wizard._sizing_phases(exp, free_bytes=FREE, source="gfs", vram_gib=12.)
    assert phases.streamed is not None
    assert phases.preprocess_backend == "cpu"
    assert phases.ingest_envelope_bytes == 0
    assert phases.host_preparation_bytes == phases.ingest.host_preprocess_bytes
    assert phases.host_preparation_bytes > phases.streamed.host_bytes
    # Refused on what it certainly holds, never on the estimate above it.
    floor = phases.host_preparation_floor_bytes
    assert 0 < floor < phases.host_preparation_bytes
    assert phases.host_preparation_refusal(floor) is None
    assert "host RAM" in phases.host_preparation_refusal(floor - 1)


@pytest.fixture
def small_host(monkeypatch):
    # A large card on a host with less RAM than the card: the planner keeps
    # the domain resident, so the CPU preparation is the only host cost.
    host = {"bytes": 16 * wizard.GIB}
    def machine(**kwargs):
        return Machine(vram_bytes=kwargs["vram_bytes"], host_bytes=host["bytes"],
                       name=kwargs["name"])
    monkeypatch.setattr(streaming, "planner_machine", machine)
    def detect(*args, **kwargs):
        pytest.fail("a declared wizard card must never probe the local GPU")
    monkeypatch.setattr(Machine, "detect", detect)
    return host


def test_cpu_preparation_larger_than_host_ram_is_refused(small_host):
    free = int(wizard.card_assumed_free_gib(48.) * wizard.GIB)
    exp = wizard.experiment_from_text(
        config(nz=76, tiles="auto", dims=[(902, 720)]), source="<resident>")
    small_host["bytes"] = 8 * wizard.GIB
    with pytest.raises(wizard.DomainFitError, match="host RAM") as refused:
        wizard._sizing_phases(exp, free_bytes=free, source="gfs", vram_gib=48.)
    assert refused.value.resource == "host"
    phases = refused.value.phases
    assert phases.streamed is None and phases.preprocess_backend == "cpu"
    assert phases.ingest_envelope_bytes == 0
    assert phases.host_preparation_floor_bytes > small_host["bytes"]
    # A host that holds the floor is not refused, even where the estimated
    # peak is over its RAM: that preparation may complete.
    small_host["bytes"] = phases.host_preparation_floor_bytes
    assert phases.host_preparation_bytes > small_host["bytes"]
    wizard._sizing_phases(exp, free_bytes=free, source="gfs", vram_gib=48.)
    small_host["bytes"] = 128 * wizard.GIB
    admitted = wizard._sizing_phases(exp, free_bytes=free, source="gfs", vram_gib=48.)
    assert admitted.host_preparation_bytes == phases.host_preparation_bytes
    line = wizard.sizing_summary(exp, admitted.forecast, free, 48., admitted)
    assert (f"preparation on the CPU holds about "
            f"{admitted.host_preparation_bytes / wizard.GIB:.2f} GiB of the "
            "host's 128.00 GiB of RAM") in line


def test_fit_never_admits_a_preparation_the_host_cannot_hold(small_host):
    free = int(wizard.card_assumed_free_gib(48.) * wizard.GIB)
    kwargs = dict(ratios=(), free_bytes=free, hours=6, start_time=START,
                  projection=wizard._projection_entries(35.3, -97.5, "auto"),
                  source="gfs", name="point", root_dx_m=3000., nz=76,
                  vram_gib=48., point_extent_km=20000.)
    small_host["bytes"] = 12 * wizard.GIB
    dims, exp = wizard.fit_ladder(**kwargs, tiles="auto")
    phases = wizard._sizing_phases(exp, free_bytes=free, source="gfs", vram_gib=48.)
    budget = wizard.sizing_budget_bytes(exp, free_bytes=free, vram_gib=48.,
                                          forcing_interval_seconds=10800.)
    host_target = small_host["bytes"] - wizard.fit_headroom_bytes(small_host["bytes"])
    assert phases.preprocess_backend == "cpu"
    # Steered on the estimated peak, which is above the refusal's floor.
    assert 0 < phases.host_preparation_floor_bytes < phases.host_preparation_bytes
    assert phases.host_preparation_bytes <= host_target
    assert phases.host_ram_bytes == small_host["bytes"]
    assert 0 < phases.peak_envelope_bytes <= budget
    # The same card on a host with room for more: RAM is what bound it.
    small_host["bytes"] = 128 * wizard.GIB
    roomy, _ = wizard.fit_ladder(**kwargs, tiles="auto")
    assert dims[0][0] * dims[0][1] < roomy[0][0] * roomy[0][1]


def test_fit_keeps_a_domain_whose_preparation_fits_the_host(small_host, monkeypatch):
    # A 902x720x76 root over 6 h peaks at 15.32e9 bytes in a real CPU
    # preparation (tests/test_cpu_preparation_host_ram.py), which a 16 GiB
    # host holds: sizing on the preparation's estimated peak keeps it.
    # The card is 64 GB so that HOST RAM is what binds whatever the
    # forecast margin: on the 48 GB card this test used until A163 the card
    # bound first at 896x718 at the first repair's 1.16 of the subtotal.
    from woof.ingest import preparation_workers

    # 850d0957d, development branch, binds the historical peak to eight workers.
    # baecc2bcf later adds scratch for wider pools. Declare the measured
    # target's CPU budget here, retaining its card, RAM and 902x720 receipt.
    monkeypatch.setattr(preparation_workers, "cpu_budget", lambda: {
        "affinity_cpus": 8, "cgroup_cpus": None, "available_cpus": 8})
    monkeypatch.setattr(preparation_workers, "host_available_bytes",
                        lambda: small_host["bytes"])
    assert preparation_workers.effective_workers() == 8
    free = int(wizard.card_assumed_free_gib(64.) * wizard.GIB)
    kwargs = dict(ratios=(), free_bytes=free, hours=6, start_time=START,
                  projection=wizard._projection_entries(35.3, -97.5, "auto"),
                  source="gfs", name="point", root_dx_m=3000., nz=76,
                  vram_gib=64., point_extent_km=20000.)
    assert small_host["bytes"] == 16 * wizard.GIB
    dims, exp = wizard.fit_ladder(**kwargs, tiles="auto")
    assert dims[0][0] * dims[0][1] >= 902 * 720
    phases = wizard._sizing_phases(exp, free_bytes=free, source="gfs", vram_gib=64.)
    host_target = small_host["bytes"] - wizard.fit_headroom_bytes(small_host["bytes"])
    assert 15_315_107_840 <= phases.host_preparation_bytes <= host_target


def test_auto_accepts_a_tree_whose_actual_plan_is_all_resident(declared_machine):
    exp = wizard.experiment_from_text(config(tiles="auto", dims=[(100, 100), (90, 90)],
                                             ratios=(3,)), source="<tree>")
    phases = wizard._sizing_phases(exp, free_bytes=FREE, source="gfs", vram_gib=12.)
    assert phases.tree_road.priced and not phases.tree_road.refusal
    assert not phases.tree_road.streams_any


def test_a_streamed_tree_is_priced_without_resident_fallback(declared_machine):
    exp = wizard.experiment_from_text(config(tiles="on", dims=[(100, 100), (90, 90)],
                                          ratios=(3,)), source="<tree>")
    phases = wizard._sizing_phases(exp, free_bytes=FREE, source="gfs", vram_gib=12.)
    assert phases.tree_road.priced and not phases.tree_road.refusal
    assert phases.tree_road.streams_any
    assert len(phases.tree_road.rows) == 2
    assert all(row["road"] == "streamed" for row in phases.tree_road.rows)


def test_missing_host_resources_refuses_without_local_gpu_probe(monkeypatch):
    monkeypatch.setattr(streaming, "planner_machine", lambda **kwargs: None)
    exp = wizard.experiment_from_text(config(tiles="auto"), source="<missing host>")
    with pytest.raises(wizard.DomainFitError, match="host RAM"):
        wizard._sizing_phases(exp, free_bytes=FREE, source="gfs", vram_gib=12.)


def test_real_host_budget_refusal_is_preserved(monkeypatch):
    monkeypatch.setattr(streaming, "planner_machine", lambda **kwargs:
                        Machine(vram_bytes=FREE, host_bytes=wizard.GIB, name="small host"))
    exp = wizard.experiment_from_text(config(tiles="on", nz=76), source="<small host>")
    with pytest.raises(wizard.DomainFitError, match="--tiles on"):
        wizard._sizing_phases(exp, free_bytes=FREE, source="gfs", vram_gib=12.)


def test_real_cli_emits_requested_levels_and_rechecks_the_same_machine(
        tmp_path, capsys, declared_machine):
    from woof.cli import main

    out = tmp_path / "area.toml"
    rc = main(["domain", "--point", "35.3,-97.5", "--source", "gfs",
               "--cycle", "2026-07-29T18", "--hours", "6", "--card", "12gb",
               "--root-dx", "3", "--nz", "76", "--tiles", "--out", str(out),
               "--explain"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    raw = tomllib.loads(out.read_text())
    assert raw["shared"]["nz"] == 76
    assert len(raw["shared"]["eta_levels"]) == 77
    assert raw["tiles"]["mode"] == "auto"
    assert "--free-gib 11.25 --vram-gib 12" in captured.out
    assert {call["vram_bytes"] for call in declared_machine} == {FREE}


def test_cli_refuses_too_few_levels_before_writing(tmp_path, capsys):
    from woof.cli import main

    out = tmp_path / "area.toml"
    rc = main(["domain", "--point", "35.3,-97.5", "--source", "gfs",
               "--cycle", "2026-07-29T18", "--card", "12gb", "--nz", "3",
               "--out", str(out)])
    assert rc == 2
    assert "--nz must be at least 4" in capsys.readouterr().err
    assert not out.exists()
