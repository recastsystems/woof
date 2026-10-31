"""The physics composer: every registry scheme listed, refusals named, presets runnable.

The catalog and the check are read straight from the engine; the GUI half
runs the real server with a runner that answers ``physics-catalog`` in
this process instead of a subprocess, so the payloads are the engine's.
"""

from __future__ import annotations

import contextlib
import io
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import physics_catalog as pc
from woof.gui.jobs import Refused, Runner, engine_argv
from woof.gui.server import build_server, serve_in_thread
from woof.physics_menu import shipped_profiles
from woof.physics_registry import physics_registry

from test_gui_server import DRAFT, FakeRunner, request


@pytest.fixture(autouse=True)
def _no_publication_probe(monkeypatch):
    # New forecast puts a recent start to the fetch's object probe before it accepts it; no server is asked in a
    # test, or a start near the publication frontier waits on the network and the page's request times out.
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    # A page asks through its own short probe; no server is asked in a test either.
    monkeypatch.setattr("woof.source_availability.quick_head", lambda url: True)


@pytest.fixture(scope="module")
def catalog():
    return pc.catalog()


def test_the_catalog_lists_every_scheme_the_registry_has(catalog):
    registry = physics_registry()
    listed = {f["id"]: sorted(s["id"] for s in f["schemes"]) for f in catalog["families"]}
    assert listed == {f: sorted(c["options"]) for f, c in registry["components"].items()}
    assert catalog["scheme_count"] == sum(len(c["options"]) for c in registry["components"].values())
    assert sorted(s["id"] for s in catalog["suites"]) == sorted(registry["templates"])


def test_every_scheme_has_words_and_every_word_row_names_a_real_scheme(catalog):
    registry = physics_registry()
    words = pc.table()["descriptions"]
    for family, component in registry["components"].items():
        assert set(words.get(family, {})) == set(component["options"]), family
    for scheme in (s for f in catalog["families"] for s in f["schemes"]):
        assert scheme["description"] and scheme["cost"]["words"]
        if scheme["implemented"]:
            assert scheme["cards"]["16gb"]["columns"], scheme["id"]


def test_every_preset_is_a_create_page_suite_the_engine_admits():
    for row in pc.presets():
        assert row["suite"] in shipped_profiles(), row["id"]
        verdict = pc.check({"preset": row["id"]})
        assert verdict["valid"] and verdict["named_suite"] == row["suite"], (row["id"], verdict.get("refusal"))


@pytest.mark.parametrize("choices, door, words", [
    ({"pbl": "myj", "surface_layer": "revised-mm5"}, "configuration", "requires sf_sfclay_physics=2"),
    ({"cumulus": "grell-freitas", "pbl": "off"}, "configuration", "requires a PBL scheme"),
    ({"radiation": "off"}, "radiation-off-land-surface", "radiation switched entirely OFF"),
    ({"microphysics": "sase"}, "registry", "Not implemented"),
])
def test_an_invalid_combination_gets_the_engines_refusal_and_valid_neighbours(choices, door, words):
    verdict = pc.check({"choices": choices})
    assert verdict["valid"] is False
    assert verdict["refusal"]["door"] == door and words in verdict["refusal"]["message"]
    assert verdict["neighbours"], choices
    for neighbour in verdict["neighbours"]:
        again = pc.check({"choices": {**choices, **neighbour["choices"]}, "settings": neighbour["settings"]})
        assert again["valid"], (neighbour, again.get("refusal"))


def test_the_nearest_repair_keeps_what_was_chosen():
    verdict = pc.check({"choices": {"pbl": "myj", "cumulus": "off", "surface_layer": "revised-mm5"}})
    assert verdict["valid"] is False
    assert verdict["neighbours"]
    # Each repair changes one of the two clashing picks and keeps the cumulus the caller chose.
    for row in verdict["neighbours"]:
        assert set(row["choices"]) <= {"pbl", "surface_layer"} and "cumulus" not in row["choices"], row


def test_a_pbl_choice_brings_the_surface_layer_it_needs():
    """U-E4: MYJ alone runs, with the Eta surface layer its registry option requires."""
    verdict = pc.check({"choices": {"pbl": "myj"}})
    assert verdict["valid"], verdict.get("refusal")
    assert verdict["resolved"]["surface_layer"] == "eta-similarity"


@pytest.mark.parametrize("suite, family, choice, follows", [
    # U-E4 (customer run_5f83d3dcf5f1cec3): each of these was refused at
    # every source and spacing because the choice left behind a part of the
    # suite that belonged to the scheme it replaced.
    ("thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1", "pbl", "myj", {"sf_sfclay_physics": 2}),
    ("wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1", "pbl", "ysu", {"sf_sfclay_physics": 1}),
    ("wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1", "pbl", "shinhong", {"sf_sfclay_physics": 1}),
    ("thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1", "microphysics", "lin-mp2",
     {"ra_rrtmg_variant": "rrtmg_legacy", "wrf_rrtmg_compatibility": "wrf-rrtmg-4-4-legacy-v1"}),
    ("thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1", "microphysics", "thompson-mp8",
     {"mp_physics": 8, "thompson_version": "wrf_461", "thompson_fork_snow_fall": "blend"}),
    ("thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1", "microphysics", "wsm6-mp6", {"mp_physics": 6}),
    ("thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1", "radiation", "rte-rrtmgp",
     {"ra_rrtmg_variant": "rte-rrtmgp"}),
    ("thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1", "radiation", "new-goddard", {"ra_lw_physics": 5}),
    ("thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1", "radiation", "wrf-rrtm-dudhia", {"ra_lw_physics": 1}),
    ("thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1", "pbl", "ysu",
     {"bl_pbl_physics": 1, "sf_sfclay_physics": 1, "bl_mynn_version": "wrf_461"}),
])
def test_one_scheme_choice_on_a_suite_runs_with_what_it_brings(suite, family, choice, follows):
    verdict = pc.check({"source": "gfs", "suite": suite, "choices": {family: choice}, "dx_km": 3})
    assert verdict["valid"], verdict.get("refusal")
    changed = verdict["changed_from_suite"]
    assert {key: changed.get(key, pc._suite_switches(suite).get(key)) for key in follows} == follows
    # The HRRR suite keeps its PBL scalar mixing: the new scheme's own
    # scalars are diffused (woof.config.SCALAR_PBLMIX_SPECIES).
    if suite.startswith("thompson-mp28-mynn-gsd41"):
        assert "scalar_pblmix" not in changed


def test_a_suites_scheme_generation_leaves_with_the_lineage_it_came_from():
    """The HRRR suite's NOAA WRF 3.9 fork Thompson does not outlive its MYNN.

    Swapping that suite's PBL to YSU (with Noah and KF) ran the fork under
    a label reading only "Thompson aerosol-aware", so a request for WRF's
    mp28 got a different Thompson generation and a quarter less rain.
    """

    suite = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
    suite_switches = pc._suite_switches(suite)
    assert suite_switches["thompson_version"] == "wrf_39_noaa"

    def resolved(choices, settings=None):
        verdict = pc.check({"source": "gfs", "suite": suite, "choices": choices,
                            "settings": settings or {}, "dx_km": 9})
        assert verdict["valid"], verdict.get("refusal")
        changed = verdict["changed_from_suite"]
        return {key: changed.get(key, suite_switches.get(key)) for key in (
            "thompson_version", "thompson_fork_snow_fall", "bl_pbl_physics", "sf_sfclay_physics",
            "wif_input_opt", "ra_rrtmg_variant", "scalar_pblmix")}

    wrf = {"pbl": "ysu", "land_surface": "noah", "cumulus": "kain-fritsch"}
    for choices in (wrf, {**wrf, "microphysics": "thompson-aerosol-mp28"}):
        mix = resolved(choices)
        assert mix["thompson_version"] == "wrf_461" and mix["thompson_fork_snow_fall"] == "blend", choices
        # The rest of what the suite set stays: WIF aerosol, legacy RRTMG, scalar PBL mixing.
        assert (mix["bl_pbl_physics"], mix["sf_sfclay_physics"], mix["wif_input_opt"],
                mix["ra_rrtmg_variant"], mix["scalar_pblmix"]) == (1, 1, 1, "rrtmg_legacy", 1), choices
    # A choice that keeps every scheme carrying the lineage keeps the fork.
    assert resolved({"cumulus": "kain-fritsch"})["thompson_version"] == "wrf_39_noaa"
    # A request that wants the fork in the mix says so, and gets it.
    asked = resolved(wrf, {"thompson_version": "wrf_39_noaa", "thompson_fork_snow_fall": "wrf_39_noaa"})
    assert asked["thompson_version"] == asked["thompson_fork_snow_fall"] == "wrf_39_noaa"


def test_a_wrf_mp28_selection_on_the_hrrr_suite_composes_wrf_461_thompson_and_says_which():
    """The selection a WRF 4.6 user made: the HRRR-lineage mp28 suite with
    YSU, Noah, Kain-Fritsch and surface layer 1 (revised MM5).  It ran HRRR's
    WRFV3.9 fork Thompson under a label that never named it."""

    suite = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
    label = physics_registry()["templates"][suite]["label"]
    assert label.startswith("HRRR WRFV3.9 aerosol Thompson"), label
    asked = {"pbl": "ysu", "land_surface": "noah", "cumulus": "kain-fritsch",
             "surface_layer": "revised-mm5"}
    mix = pc.check({"source": "gfs", "suite": suite, "choices": asked, "dx_km": 9})
    assert mix["valid"], mix.get("refusal")
    toml = tomllib.loads(mix["experiment_toml"])
    shared = toml.get("shared", toml)
    assert shared["thompson_version"] == "wrf_461" and shared["sf_sfclay_physics"] == 1
    assert "WRFV3.9" not in mix["words"]
    # The untouched suite composes exactly its own settings, fork included,
    # and its words say which Thompson it runs.
    kept = pc.check({"source": "gfs", "suite": suite, "dx_km": 9})
    assert kept["valid"] and kept["changed_from_suite"] == {}
    shared = tomllib.loads(kept["experiment_toml"])
    shared = shared.get("shared", shared)
    assert shared["thompson_version"] == shared["thompson_fork_snow_fall"] == "wrf_39_noaa"
    assert "HRRR's WRFV3.9 version" in kept["words"]
    assert kept["named_suite_label"] == label


def test_a_name_the_registry_does_not_have_is_refused_in_plain_words():
    with pytest.raises(pc.CatalogError, match="no scheme called 'nope'"):
        pc.check({"choices": {"pbl": "nope"}})


def test_the_run_manifest_records_the_stated_suite():
    from woof.runplan import manifest_physics

    plan = SimpleNamespace(config_intent={"physics_profile": pc.presets()[0]["suite"]},
                           run_options={}, config_kind="intent")
    recorded = manifest_physics(plan)
    assert recorded["suite"] == pc.presets()[0]["suite"] and recorded["stated_by"] == "plan"
    assert recorded["components"]["microphysics"] and recorded["switches"]["mp_physics"]
    unstated = manifest_physics(SimpleNamespace(config_intent={}, run_options={}, config_kind="intent"))
    assert unstated["suite"] is None


class CatalogRunner(FakeRunner):
    """Answers ``physics-catalog`` in this process with the engine's own output."""

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        if "physics-catalog" not in argv:
            return super().query(argv, cwd=cwd, timeout=timeout, log=log)
        self.queries.append(list(argv))
        tail = argv[argv.index("physics-catalog") + 1:]
        args = SimpleNamespace(json="--json" in tail, check=None, preset=None, source=None)
        for flag in ("check", "preset", "source"):
            if f"--{flag}" in tail:
                setattr(args, flag, tail[tail.index(f"--{flag}") + 1])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = pc.main(args)
        document = json.loads(out.getvalue())
        if code != 0:
            # What Runner.query does with a nonzero exit.
            raise Refused(str(document.get("error") or f"The command exited {code}."), document)
        return document


@pytest.fixture()
def gui(tmp_path):
    runner = CatalogRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    yield server, runner
    server.shutdown()
    server.server_close()


def test_the_gui_serves_the_catalog_and_the_check(gui):
    server, _ = gui
    response, body = request(server, "GET", "/api/physics", timeout=60)
    assert response.status == 200 and body["scheme_count"] == pc.catalog()["scheme_count"]
    assert "physics-catalog --json" in body["command"]
    response, body = request(server, "POST", "/api/physics/check",
                             body={"choices": {"pbl": "myj", "surface_layer": "revised-mm5"}}, timeout=60)
    assert response.status == 200 and body["check"]["valid"] is False
    assert body["check"]["neighbours"]
    response, body = request(server, "POST", "/api/physics/check", body={"choices": {"pbl": "nope"}}, timeout=60)
    assert response.status == 400 and "nope" in body["message"]


def test_a_preset_on_create_states_its_suite_and_spacing_in_the_plan(gui):
    server, _ = gui
    row = next(r for r in pc.presets() if r["dx_km"] >= 0.5)
    draft = {**DRAFT, "dx_km": None, "preset": row["id"], "dry_run": True}
    response, body = request(server, "POST", "/api/create/start", body=draft, timeout=60)
    assert response.status == 200, body
    intent = body["plan"]["config"]["intent"]
    assert intent["physics_profile"] == row["suite"] and intent["root_dx_km"] == row["dx_km"]
    small = next((r for r in pc.presets() if r["dx_km"] < 0.5), None)
    if small is not None:
        response, body = request(server, "POST", "/api/create/start",
                                 body={**draft, "preset": small["id"]}, timeout=60)
        assert response.status == 400 and small["id"] in body["message"]


def test_radiation_by_stream_is_composed_as_the_capability_door_composes_it():
    verdict = pc.check({"choices": {"radiation": {"longwave": 4, "shortwave": 1}}})
    assert verdict["valid"], verdict.get("refusal")
    assert verdict["resolved"]["radiation"] == "independent-4-1"
    too_deep = pc.check({"nz": 200})
    assert not too_deep["valid"] and "\n" not in too_deep["words"]


def test_the_real_runner_turns_an_unknown_name_into_the_engines_sentence(tmp_path):
    # The real subprocess, because the in-process runner above cannot show
    # what an exit code does to the answer.
    argv = engine_argv("physics-catalog", "--json", "--check", json.dumps({"choices": {"pbl": "nope"}}))
    with pytest.raises(Refused) as caught:
        Runner(tmp_path).query(argv, timeout=300)
    assert "pbl has no scheme called 'nope'" in str(caught.value)
    assert caught.value.document["error"] == str(caught.value)


def test_the_gui_answers_unknown_names_and_sources_with_a_400(gui):
    server, _ = gui
    for body in ({"preset": "bogus"}, {"source": "nosuch"}, {"choices": {"pbl": "nope"}},
                 {"settings": {"mp_physcs": 50}}):
        response, answer = request(server, "POST", "/api/physics/check", body=body, timeout=60)
        assert response.status == 400, (body, answer)
    response, answer = request(server, "GET", "/api/physics?source=nosuch", timeout=60)
    assert response.status == 400 and "nosuch" in answer["message"] and answer["fix"]


def test_an_unknown_source_is_refused_not_replaced_by_the_default():
    with pytest.raises(pc.CatalogError, match="No data source called 'nosuch'"):
        pc.catalog(source="nosuch")
    with pytest.raises(pc.CatalogError, match="No data source called 'nosuch'"):
        pc.check({"source": "nosuch"})


def test_a_day_only_suite_says_so_and_a_window_with_night_is_refused():
    day_only = next(r for r in pc.presets() if pc.check({"preset": r["id"]}).get("day_only_reason"))
    verdict = pc.check({"preset": day_only["id"]})
    assert verdict["valid"] and verdict["words"].startswith("Runs by day only")
    night = pc.check({"preset": day_only["id"], "cycle": "2026-06-21T06", "hours": 6, "lat": 35, "lon": -97})
    assert not night["valid"] and night["refusal"]["door"] == "nocturnal-radiation"
    assert night["neighbours"] and all(n["named_suite"] is None or not pc.check(
        {"suite": n["named_suite"]}).get("day_only_reason") for n in night["neighbours"])
    daylight = pc.check({"preset": day_only["id"], "cycle": "2026-06-21T15", "hours": 3, "lat": 35, "lon": -97})
    assert daylight["valid"] and "all daylight" in daylight["words"]
    with pytest.raises(pc.CatalogError, match="cycle, lat and lon together"):
        pc.check({"cycle": "2026-06-21T15"})


def test_a_mix_no_suite_names_comes_with_the_lines_that_run_it():
    import tomllib

    verdict = pc.check({"choices": {"microphysics": "p3-mp50"}, "dx_km": 12})
    assert verdict["valid"] and verdict["named_suite"] is None
    assert verdict["words"] == "Runs. No named set matches these choices."
    assert verdict["named_suite_label"] is None
    shared = tomllib.loads(verdict["experiment_toml"])["shared"]
    assert shared["mp_physics"] == 50
    again = pc.check({"settings": shared, "dx_km": 12})
    assert again["valid"] and again["resolved"] == verdict["resolved"]
    assert verdict["changed_from_suite"] == {"mp_physics": 50}


def test_a_named_set_is_said_by_its_plain_name_not_its_id():
    # One scheme changed from the default lands on another registered set, which the words name by its label.
    verdict = pc.check({"choices": {"microphysics": "nssl2-mp18"}, "dx_km": 12})
    named = verdict["named_suite"]
    assert verdict["valid"] and named and named != pc.default_suite(), verdict.get("refusal")
    label = physics_registry()["templates"][named]["label"]
    assert verdict["named_suite_label"] == label and label != named
    assert label in verdict["words"] and named not in verdict["words"]


def test_the_mix_goes_into_an_experiment_file_the_loader_accepts(tmp_path):
    import tomllib

    from woof.experiment import load_experiment

    source = Path(__file__).resolve().parents[1] / "configs" / "aifs_single_mesoscale_demo.toml"
    text = pc.apply_to_experiment(source.read_text(encoding="utf-8"),
                                  {"choices": {"microphysics": "p3-mp50", "cumulus": "off"}})
    written = tmp_path / "mix.toml"
    written.write_text(text, encoding="utf-8")
    run = load_experiment(written).domains[0].run
    assert (run.mp_physics, run.cu_physics) == (50, 0)
    before = tomllib.loads(source.read_text(encoding="utf-8"))
    after = tomllib.loads(text)
    assert {k for k in after["shared"] if after["shared"][k] != before["shared"].get(k)} == {"mp_physics"}
    with pytest.raises(pc.CatalogError, match="requires sf_sfclay_physics=2"):
        pc.apply_to_experiment(source.read_text(encoding="utf-8"),
                               {"choices": {"pbl": "myj", "surface_layer": "revised-mm5"}})


def test_a_run_from_an_experiment_file_records_the_files_physics(tmp_path):
    from woof.runplan import manifest_physics

    source = Path(__file__).resolve().parents[1] / "configs" / "aifs_single_mesoscale_demo.toml"
    text = pc.apply_to_experiment(source.read_text(encoding="utf-8"),
                                  {"choices": {"microphysics": "p3-mp50"}})
    plan = SimpleNamespace(config_intent=None, run_options={}, config_kind="inline",
                           config_bytes=lambda: text.encode("utf-8"))
    recorded = manifest_physics(plan)
    assert recorded["switches"]["mp_physics"] == 50 and recorded["components"]["microphysics"] == "p3-mp50"
    assert recorded["suite"] is None and recorded["domains"][0]["cu_physics"] == 1
    # Switches only a scheme that is not running reads are set apart.
    assert {"wsm6_hail_opt", "morr_rimed_ice"} <= set(recorded["inert"])
    assert not {"wsm6_hail_opt", "morr_rimed_ice"} & set(recorded["switches"])
    assert "inline configuration" in recorded["stated_by"]


def test_a_mix_written_under_a_new_name_takes_its_companion_namelist(tmp_path):
    import shutil

    configs = Path(__file__).resolve().parents[1] / "configs"
    shutil.copyfile(configs / "aifs_single_mesoscale_demo.toml", tmp_path / "exp.toml")
    shutil.copyfile(configs / "aifs_single_mesoscale_demo.namelist.wps", tmp_path / "exp.namelist.wps")
    written = pc.write_experiment(tmp_path / "exp.toml", tmp_path / "mix.toml",
                                  {"choices": {"microphysics": "p3-mp50"}})
    assert [p.name for p in written] == ["mix.toml", "mix.namelist.wps"]
    assert (tmp_path / "mix.namelist.wps").read_bytes() == (tmp_path / "exp.namelist.wps").read_bytes()
    (tmp_path / "taken.namelist.wps").write_text("someone else's\n")
    with pytest.raises(pc.CatalogError, match="taken.namelist.wps already exists"):
        pc.write_experiment(tmp_path / "exp.toml", tmp_path / "taken.toml", {"choices": {"microphysics": "p3-mp50"}})
    assert not (tmp_path / "taken.toml").exists()
    assert (tmp_path / "taken.namelist.wps").read_text() == "someone else's\n"
    (tmp_path / "exp.namelist.input").write_text("&physics\n/\n")
    with pytest.raises(pc.CatalogError, match="states its physics as well"):
        pc.write_experiment(tmp_path / "exp.toml", tmp_path / "other.toml", {"choices": {"microphysics": "p3-mp50"}})


def test_the_check_runs_the_cumulus_the_emitter_writes_at_that_spacing():
    import tomllib

    from woof.domain_wizard import cumulus_by_domain

    suite = pc.default_suite()
    for dx_km in (3.0, 12.0):
        emitted = cumulus_by_domain([(60, 60)], (), profile=suite, root_dx_m=dx_km * 1000.0)[0]
        verdict = pc.check({"dx_km": dx_km})
        assert verdict["valid"] and verdict["named_suite"] == suite
        shared = tomllib.loads(verdict["experiment_toml"])["shared"]
        assert shared["cu_physics"] == emitted
        assert verdict["cost"]["words"] == "The default's cost."
    fine = pc.check({"dx_km": 3})
    assert fine["resolved"]["cumulus"] == "off" and fine["choices"]["cumulus"] == "off"
    assert not [a for a in fine["advisories"] if a["headline"].startswith("CUMULUS")]
    assert fine["cumulus_retired"] and fine["words"].startswith("Runs. Cumulus is off at 3 km")
    assert "cu_physics" not in fine["words"] and "root" not in fine["words"]
    assert fine["changed_from_suite"] == {"cu_physics": 0, "cudt_minutes": 0.0}
    coarse = pc.check({"dx_km": 12})
    assert coarse["resolved"]["cumulus"] == "kain-fritsch" and coarse["cumulus_retired"] is None
    # A mix on the default suite is retired the same way.
    mix = pc.check({"dx_km": 3, "choices": {"microphysics": "p3-mp50"}})
    assert mix["resolved"]["cumulus"] == "off" and mix["cumulus_retired"]
    # Naming the suite, or choosing cumulus, keeps it, as --physics-profile
    # and Create's physics_profile do; the advisory then says what it costs.
    for request in ({"dx_km": 3, "suite": suite}, {"dx_km": 3, "choices": {"cumulus": "kain-fritsch"}},
                    {"dx_km": 3, "settings": {"cu_physics": 1}}):
        kept = pc.check(request)
        assert kept["resolved"]["cumulus"] == "kain-fritsch" and kept["cumulus_retired"] is None
        assert [a for a in kept["advisories"] if a["headline"].startswith("CUMULUS")]
    assert cumulus_by_domain([(60, 60)], (), profile=suite, root_dx_m=3000.0, cumulus_requested=True) == [1]


def _fine_demo(cumulus: int | None = None) -> str:
    source = Path(__file__).resolve().parents[1] / "configs" / "aifs_single_mesoscale_demo.toml"
    text = source.read_text(encoding="utf-8")
    assert "dx = 12000.0" in text and "cu_physics = 1" in text
    fine = text.replace("dx = 12000.0", "dx = 3000.0").replace("dy = 12000.0", "dy = 3000.0")
    if cumulus is not None:
        fine = fine.replace("cu_physics = 1", f"cu_physics = {cumulus}")
        fine = fine.replace("cudt_minutes = 5.0", "cudt_minutes = 0.0" if not cumulus else "cudt_minutes = 5.0")
    return fine


def test_a_mix_into_a_fine_root_keeps_the_cumulus_the_file_runs(tmp_path):
    # The emitter writes an active cumulus on a 3 km root only when the user
    # asked for it, so a 3 km file running cu_physics 1 records that choice;
    # a mix that touches only microphysics keeps it.
    from woof.experiment import load_experiment

    fine = _fine_demo()
    written = tmp_path / "mix.toml"
    text = pc.apply_to_experiment(fine, {"choices": {"microphysics": "p3-mp50"}})
    written.write_text(text, encoding="utf-8")
    run = load_experiment(written).domains[0].run
    assert (run.mp_physics, run.cu_physics) == (50, 1)
    assert pc.file_cumulus(text) == (1, 5.0)
    assert pc.cumulus_change(fine, text) is None


def test_a_mix_into_a_fine_root_running_no_cumulus_stays_without(tmp_path):
    from woof.experiment import load_experiment

    fine = _fine_demo(cumulus=0)
    written = tmp_path / "mix.toml"
    text = pc.apply_to_experiment(fine, {"choices": {"microphysics": "p3-mp50"}})
    written.write_text(text, encoding="utf-8")
    run = load_experiment(written).domains[0].run
    assert (run.mp_physics, run.cu_physics) == (50, 0)


def test_a_mix_that_turns_cumulus_off_says_so(tmp_path, capsys):
    fine = tmp_path / "fine.toml"
    fine.write_text(_fine_demo(), encoding="utf-8")
    out = tmp_path / "off.toml"
    request = json.dumps({"choices": {"microphysics": "p3-mp50", "cumulus": "off"}})
    args = SimpleNamespace(json=False, check=request, preset=None, source=None, emit=False,
                           into=str(fine), out=str(out))
    assert pc.main(args) == 0
    printed = capsys.readouterr().out
    assert "wrote" in printed and "turns the root's cumulus off (cu_physics 1 -> 0)" in printed
    assert pc.file_cumulus(out.read_text(encoding="utf-8"))[0] == 0


def _fitted(tmp_path, edit=None, spelling="quoted"):
    """A nested file as ``woof domain-fit`` writes it (quoted table and key names), or spelled bare."""

    from woof import starter_template
    from woof.toml_document import emit_experiment_toml
    from test_starter_template import starter

    path, raw = starter(tmp_path, nested=True)
    if edit is not None:
        edit(raw)
    write = starter_template.render_tables if spelling == "quoted" else emit_experiment_toml
    path.write_text(write(raw), encoding="utf-8")
    assert ('["shared"]' in path.read_text(encoding="utf-8")) == (spelling == "quoted")
    return path


def _runs(path, key):
    from woof.experiment import load_experiment

    return [getattr(domain.run, key) for domain in load_experiment(path).domains]


@pytest.mark.parametrize("spelling", ["quoted", "bare"])
def test_a_mix_goes_into_a_fitted_file_and_changes_only_its_own_line(tmp_path, capsys, spelling):
    fitted = _fitted(tmp_path, spelling=spelling)
    before = fitted.read_text(encoding="utf-8")
    key = '"{}"'.format if spelling == "quoted" else "{}".format
    assert f'{key("mp_physics")} = 10' in before
    # A value written over several lines is one statement: the switch the
    # mix leaves alone keeps the file's own lines, and none is cut in two.
    assert f'{key("ra_rrtmg_variant")} = "rte-rrtmgp"' in before
    before = before.replace(f'{key("ra_rrtmg_variant")} = "rte-rrtmgp"',
                            f'{key("ra_rrtmg_variant")} = """\nrte-rrtmgp"""')
    fitted.write_text(before, encoding="utf-8")
    out = tmp_path / "mix.toml"
    args = SimpleNamespace(json=False, check=json.dumps({"choices": {"microphysics": "p3-mp50"}}), preset=None,
                           source=None, emit=False, into=str(fitted), out=str(out))
    assert pc.main(args) == 0, capsys.readouterr().err
    after = out.read_text(encoding="utf-8")
    assert _runs(out, "mp_physics") == [50, 50] and _runs(out, "cu_physics") == [1, 0]
    # The one line the mix changes, and nothing else of the file.
    assert len(after.splitlines()) == len(before.splitlines())
    assert [(old, new) for old, new in zip(before.splitlines(), after.splitlines()) if old != new] == \
           [(f'{key("mp_physics")} = 10', "mp_physics = 50")]


@pytest.mark.parametrize("spelling", ["quoted", "bare"])
def test_a_switch_only_some_grids_state_reaches_the_grids_that_do_not(tmp_path, spelling):
    import tomllib

    def root_only(raw):
        raw["shared"].pop("radt", None)
        raw["domain"][0]["radt"] = 12.0
        raw["domain"][1].pop("radt", None)

    fitted = _fitted(tmp_path, root_only, spelling)
    assert _runs(fitted, "radt") == [12.0, 0.0]
    mixed = tmp_path / "mixed.toml"
    mixed.write_text(pc.apply_to_experiment(fitted.read_text(encoding="utf-8"), {"settings": {"radt": 15.0}}),
                     encoding="utf-8")
    assert _runs(mixed, "radt") == [15.0, 15.0]

    def every_grid(raw):
        raw["shared"].pop("radt", None)
        raw["domain"][0]["radt"] = 12.0
        raw["domain"][1]["radt"] = 6.0

    fitted = _fitted(tmp_path, every_grid, spelling)
    text = pc.apply_to_experiment(fitted.read_text(encoding="utf-8"), {"choices": {"microphysics": "p3-mp50"}})
    mixed.write_text(text, encoding="utf-8")
    assert _runs(mixed, "radt") == [12.0, 6.0]
    assert _runs(mixed, "mp_physics") == [50, 50]
    assert "radt" not in tomllib.loads(text)["shared"]


@pytest.mark.parametrize("spelling", ["quoted", "bare"])
@pytest.mark.parametrize("shared_states_it", [False, True])
def test_a_nest_stating_no_cumulus_keeps_it_off_when_the_root_gets_one(tmp_path, shared_states_it, spelling):
    def nest_without_cumulus(raw):
        raw["domain"][1].pop("cu_physics")
        if shared_states_it:
            # The root reads [shared]'s 0 too; the mix turns it on there.
            raw["shared"]["cu_physics"] = 0
            raw["domain"][0].pop("cu_physics")
        else:
            raw["shared"].pop("cu_physics", None)

    fitted = _fitted(tmp_path, nest_without_cumulus, spelling)
    before = fitted.read_text(encoding="utf-8")
    assert _runs(fitted, "cu_physics") == [0 if shared_states_it else 1, 0]
    mixed = tmp_path / "mixed.toml"
    mixed.write_text(pc.apply_to_experiment(before, {"choices": {"cumulus": "kain-fritsch"}}), encoding="utf-8")
    assert _runs(mixed, "cu_physics") == [1, 0]


def test_a_misspelled_setting_is_refused_by_name_and_nothing_is_written(tmp_path):
    with pytest.raises(pc.CatalogError, match=r"No setting called 'mp_physcs'; did you mean mp_physics\?"):
        pc.check({"settings": {"mp_physcs": 50}})
    with pytest.raises(pc.CatalogError, match="No setting called 'not_a_switch'[.]"):
        pc.check({"settings": {"not_a_switch": 1, "mp_physics": 50}})
    fitted = _fitted(tmp_path)
    out = tmp_path / "mix.toml"
    with pytest.raises(pc.CatalogError, match="mp_physcs"):
        pc.write_experiment(fitted, out, {"settings": {"mp_physcs": 50}})
    assert not out.exists()
    spelled = pc.check({"settings": {"mp_physics": 50}})
    assert spelled["valid"] and spelled["changed_from_suite"]["mp_physics"] == 50


def test_every_preset_runs_from_create_and_a_day_only_one_is_asked_the_night_door(gui):
    from woof.gui.api import DX_MIN_KM

    server, runner = gui
    # A preset Create cannot start is a promise the page does not keep.
    assert all(float(row["dx_km"]) >= DX_MIN_KM for row in pc.presets())
    day_only = next(r for r in pc.presets() if pc.check({"preset": r["id"]}).get("day_only_reason"))
    night = {**DRAFT, "dx_km": None, "preset": day_only["id"], "cycle": "2026-06-21T06", "hours": 6,
             "lat": 35.0, "lon": -97.0}
    for path, body in (("/api/create/fit", night), ("/api/create/start", {**night, "name": "night1"})):
        response, answer = request(server, "POST", path, body=body, timeout=120)
        assert response.status == 422, answer
        assert "local night" in answer["message"] and "daylight" in answer["fix"]
    assert not runner.launched and not (server.api.root / "night1").exists()
    day = {**night, "cycle": "2026-06-21T18", "hours": 3}
    response, answer = request(server, "POST", "/api/create/fit", body=day, timeout=120)
    assert response.status == 200, answer
    response, answer = request(server, "POST", "/api/create/start", body={**day, "name": "day1"}, timeout=120)
    assert response.status in (200, 202), answer
    assert runner.launched


class SuiteRunner(CatalogRunner):
    """The catalog runner, with the source offering every shipped suite as New forecast lists them."""

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        if "--physics-profiles" in argv:
            suites = sorted(shipped_profiles())
            return {"sources": [{"source_id": "gfs", "default_profile_id": suites[0],
                                 "profiles": [{"profile_id": s, "admissible": True, "is_default": i == 0}
                                              for i, s in enumerate(suites)]}],
                    "profiles": [{"profile_id": s, "summary": s} for s in suites]}
        return super().query(argv, cwd=cwd, timeout=timeout, log=log)


@pytest.fixture()
def composer(tmp_path):
    runner = SuiteRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    yield server, runner
    server.shutdown()
    server.server_close()


def test_a_composed_physics_choice_lands_in_the_plan_and_beside_it(composer):
    server, runner = composer
    body = {**DRAFT, "name": "mix1", "dx_km": 3, "physics_choices": {"microphysics": "thompson-mp8"}}
    expected = pc.check({"choices": {"microphysics": "thompson-mp8"}, "dx_km": 3, "source": "gfs"})["named_suite"]
    response, answer = request(server, "POST", "/api/create/start", body={**body, "dry_run": True}, timeout=120)
    assert response.status == 200, answer
    assert answer["plan"]["config"]["intent"]["physics_profile"] == expected
    assert answer["physics"]["suite"] == expected and answer["physics"]["resolved"]["microphysics"] == "thompson-mp8"
    assert not (server.api.root / "mix1").exists()
    response, answer = request(server, "POST", "/api/create/start", body=body, timeout=120)
    assert response.status == 200, answer
    rundir = server.api.root / "mix1"
    plan = json.loads((rundir / "plan.json").read_text(encoding="utf-8"))
    kept = json.loads((rundir / "gui-physics.json").read_text(encoding="utf-8"))
    assert plan["config"]["intent"]["physics_profile"] == expected == kept["suite"]
    assert kept["choices"] == {"microphysics": "thompson-mp8"}
    # The run's article names the suite and its schemes, citing the file the choice was kept in.
    response, article = request(server, "GET", "/api/wiki/run/mix1", timeout=60)
    assert response.status == 200, article
    fact = next(f for f in article["facts"] if f["id"] == "physics")
    assert kept["label"] in fact["text"] and "microphysics thompson-mp8" in fact["text"]
    assert "run-file:gui-physics.json" in fact["cite"]


def test_a_composed_set_that_does_not_run_is_refused_before_anything_is_written(composer):
    server, runner = composer
    refused = {**DRAFT, "name": "mix2", "physics_choices": {"pbl": "myj", "surface_layer": "revised-mm5"}}
    response, answer = request(server, "POST", "/api/create/start", body=refused, timeout=120)
    assert response.status == 422 and "sf_sfclay_physics=2" in answer["message"], answer
    assert "Physics step" in answer["fix"]
    assert not runner.launched
    assert not (server.api.root / "mix2").exists()


THOMPSON_MYJ_ETA = {"microphysics": "thompson-mp8", "pbl": "myj", "surface_layer": "eta-similarity"}


def test_a_mix_no_named_set_matches_starts_with_its_schemes_in_the_experiment(composer, tmp_path):
    """GS-02: the composer said "Runs" and New forecast could start only the mixes equal to a named set.

    Of 80 two-pick GFS mixes the engine ran 63 and the page could start 5, and MYJ could never be started from
    GFS: alone it needs the Eta surface layer, and with it the mix matched no set.  A mix the check calls valid
    now goes into the plan as its schemes and the wizard writes them into the experiment.
    """

    server, runner = composer
    place = {"source": "gfs", "cycle": "2026-09-24T18", "hours": 1, "lat": 35.5, "lon": -97.5, "dx_km": 3}
    response, answer = request(server, "POST", "/api/physics/check",
                               body={**place, "choices": THOMPSON_MYJ_ETA}, timeout=120)
    assert response.status == 200, answer
    assert answer["check"]["valid"] and answer["check"]["named_suite"] is None, answer["check"]
    # The check says what a plan carries to run it, and that New forecast starts it.
    assert answer["check"]["plan_intent"] == {"physics_choices": THOMPSON_MYJ_ETA}
    assert answer["check"]["on_create_page"] is True
    body = {**DRAFT, **place, "name": "mix3", "width_km": 300, "height_km": 300,
            "physics_choices": THOMPSON_MYJ_ETA}
    response, answer = request(server, "POST", "/api/create/start", body={**body, "dry_run": True}, timeout=120)
    assert response.status == 200, answer
    intent = answer["plan"]["config"]["intent"]
    assert intent["physics_choices"] == THOMPSON_MYJ_ETA
    assert "physics_profile" not in intent and "cumulus" not in intent
    assert answer["physics"]["suite"] is None and answer["physics"]["choices"] == THOMPSON_MYJ_ETA
    exp = _resolved_plan(tmp_path, answer["plan"], body)
    root = exp.domains[0].run
    assert (root.mp_physics, root.bl_pbl_physics, root.sf_sfclay_physics) == (8, 2, 2)
    # 3 km resolves storms, as the check said: the source's default set's cumulus is off here.
    assert root.cu_physics == 0 and answer["physics"]["resolved"]["cumulus"] == "off"
    assert not (server.api.root / "mix3").exists()
    response, answer = request(server, "POST", "/api/create/start", body=body, timeout=120)
    assert response.status == 200, answer
    rundir = server.api.root / "mix3"
    plan = json.loads((rundir / "plan.json").read_text(encoding="utf-8"))
    kept = json.loads((rundir / "gui-physics.json").read_text(encoding="utf-8"))
    assert plan["config"]["intent"]["physics_choices"] == kept["choices"] == THOMPSON_MYJ_ETA
    assert kept["suite"] is None and kept["base_suite"] == pc.default_suite("gfs")
    response, article = request(server, "GET", "/api/wiki/run/mix3", timeout=60)
    fact = next(f for f in article["facts"] if f["id"] == "physics")
    assert fact["text"].startswith("Picked schemes, no named set") and "pbl myj" in fact["text"]


def test_a_mix_is_sized_as_it_runs_and_a_refused_one_as_the_forms_set(composer):
    server, runner = composer
    sized, answer_resolve = [], runner.query

    def query(argv, *, cwd=None, timeout=0, log=None):
        if "--resolve" in argv:
            sized.append(json.loads(Path(argv[argv.index("--resolve") - 1]).read_text(encoding="utf-8")))
        return answer_resolve(argv, cwd=cwd, timeout=timeout, log=log)

    runner.query = query
    body = {**DRAFT, "dx_km": 3, "physics_choices": THOMPSON_MYJ_ETA}
    response, answer = request(server, "POST", "/api/create/fit", body=body, timeout=120)
    assert response.status == 200, answer
    assert sized[-1]["config"]["intent"]["physics_choices"] == THOMPSON_MYJ_ETA
    response, answer = request(server, "POST", "/api/create/fit",
                               body={**body, "physics_choices": {"pbl": "myj", "surface_layer": "revised-mm5"}},
                               timeout=120)
    assert response.status == 200, answer
    assert "physics_choices" not in sized[-1]["config"]["intent"]


def test_a_run_without_a_composed_choice_names_its_plans_suite_or_the_source_default(tmp_path):
    from woof.gui.wiki import Wiki

    root = tmp_path / "runs"
    for name, intent in (("named", {"source": "gfs", "physics_profile": "some-suite-v1"}), ("plain", {"source": "gfs"})):
        rundir = root / name
        rundir.mkdir(parents=True)
        (rundir / "plan.json").write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "config": {"intent": intent}}),
                                          encoding="utf-8")
    wiki = Wiki(root)
    named = {f["id"]: f["text"] for f in wiki.run_page("named", root / "named")["facts"]}
    plain = {f["id"]: f["text"] for f in wiki.run_page("plain", root / "plain")["facts"]}
    assert named["physics"] == "some-suite-v1"
    assert plain["physics"] == "The source's default set"


def _resolved_plan(tmp_path, plan: dict, draft: dict):
    """The configuration a plan New forecast wrote would run, loaded through the wizard like run-plan does."""

    from woof.gui.api import region_polygon
    from woof.runplan import load_plan, resolve_plan

    rundir = tmp_path / "resolve"
    rundir.mkdir()
    (rundir / "region.geojson").write_text(json.dumps(region_polygon(
        draft["lat"], draft["lon"], draft["width_km"], draft["height_km"])), encoding="utf-8")
    plan = {**plan, "output_root": str(rundir), "config": {"intent": {
        **plan["config"]["intent"], "polygon": str(rundir / "region.geojson")}}}
    (rundir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    _, exp, _ = resolve_plan(load_plan(rundir / "plan.json"), require_inputs=False)
    return exp


@pytest.mark.parametrize("choices, cumulus", [
    ({"microphysics": "nssl2-mp18"}, 0),
    ({"microphysics": "nssl2-mp18", "cumulus": "kain-fritsch"}, 1),
])
def test_the_run_gets_the_cumulus_the_physics_step_showed(composer, tmp_path, choices, cumulus):
    """A 3 km root: picking only microphysics shows cumulus off and runs none; picking KF shows it and runs it.

    The suite a microphysics pick matches carries Kain-Fritsch, and naming a suite used to mean asking for its
    cumulus, so the plan ran KF at 3 km while the step said cumulus was off.
    """

    server, runner = composer
    body = {**DRAFT, "name": "cu1", "dx_km": 3, "lat": 35.5, "lon": -97.5, "width_km": 300, "height_km": 300,
            "cycle": "2026-09-24T18", "hours": 3, "physics_choices": choices}
    verdict = pc.check({"choices": choices, "dx_km": 3, "source": "gfs"})
    assert verdict["valid"] and verdict["named_suite"]
    shown = verdict["resolved"]["cumulus"]
    assert (shown == "off") == (cumulus == 0)
    assert ("cumulus off" in verdict["named_suite_label"]) == (cumulus == 0)
    assert ("KF" in verdict["named_suite_label"]) == (cumulus == 1)
    response, answer = request(server, "POST", "/api/create/start", body={**body, "dry_run": True}, timeout=120)
    assert response.status == 200, answer
    intent = answer["plan"]["config"]["intent"]
    assert intent["physics_profile"] == verdict["named_suite"]
    assert intent.get("cumulus") == (None if cumulus else "grid")
    assert answer["physics"]["resolved"]["cumulus"] == shown
    exp = _resolved_plan(tmp_path, answer["plan"], body)
    assert [d.run.cu_physics for d in exp.domains] == [cumulus] * len(exp.domains)
    assert exp.domains[0].run.mp_physics == 18


def test_the_wizard_leaves_a_named_suites_cumulus_to_the_grid_only_when_told(tmp_path):
    from woof.domain_wizard import cumulus_requested_by

    named = SimpleNamespace(physics_profile="some-suite", cumulus=None)
    assert cumulus_requested_by(named)
    assert not cumulus_requested_by(SimpleNamespace(physics_profile="some-suite", cumulus="grid"))
    assert not cumulus_requested_by(SimpleNamespace(physics_profile=None, cumulus=None))
    assert cumulus_requested_by(SimpleNamespace(physics_profile=None, cumulus="suite"))


def test_a_plan_leaving_cumulus_to_the_grid_does_not_assert_the_suite_it_departs_from(tmp_path):
    """The staged and HRRR chains hand the plan's suite to their stages as an exactness assertion."""

    from woof.runplan import _asserted_profile, build_plan, generate_intent_config

    suite = pc.check({"choices": {"microphysics": "nssl2-mp18"}, "dx_km": 3, "source": "gfs"})["named_suite"]
    polygon = tmp_path / "region.geojson"
    from woof.gui.api import region_polygon

    polygon.write_text(json.dumps(region_polygon(35.5, -97.5, 300, 300)), encoding="utf-8")
    for dx, cumulus, asserted in ((3, "grid", None), (12, "grid", suite), (3, None, suite)):
        intent = {"polygon": str(polygon), "source": "gfs", "cycle": "2026-09-24T18", "hours": 3,
                  "card": "16gb", "root_dx_km": dx, "physics_profile": suite}
        if cumulus:
            intent["cumulus"] = cumulus
        plan = build_plan({"schema": "gpuwm.run-plan.v1", "name": "p", "route": "prepared",
                           "config": {"intent": intent}, "output_root": str(tmp_path / f"out{dx}{cumulus}")},
                          source="test", base_dir=tmp_path, sha256="0" * 64)
        config, _ = generate_intent_config(plan, destination=tmp_path / f"gen{dx}{cumulus}")
        assert _asserted_profile(plan, config_path=config) == asserted, (dx, cumulus)


def test_the_physics_steps_advisories_are_plain_sentences():
    for request_, name in (({"choices": {"cumulus": "kain-fritsch"}, "dx_km": 3}, "Kain-Fritsch"),
                           ({"choices": {"cumulus": "kain-fritsch"}, "dx_km": 6}, "Kain-Fritsch"),
                           ({"choices": {"cumulus": "grell-freitas"}, "dx_km": 3}, "Grell-Freitas"),
                           # Below 1 km the picks change the sub-km default, whose boundary layer is MYNN.
                           ({"choices": {"microphysics": "nssl2-mp18"}, "dx_km": 0.5}, "MYNN PBL")):
        verdict = pc.check({**request_, "source": "gfs"})
        assert verdict["advisories"], request_
        for row in verdict["advisories"]:
            words = row["words"]
            assert name in words and words.endswith(".") and words.startswith("At "), words
            assert "_physics" not in words and "domain(s)" not in words
            assert not any(tag in words for tag in ("CUMULUS", "GRAY ZONE")), words


def test_a_suite_with_no_cumulus_of_its_own_is_named_as_the_registry_names_it():
    verdict = pc.check({"choices": {"microphysics": "thompson-mp8"}, "dx_km": 3, "source": "gfs"})
    suite = verdict["named_suite"]
    assert physics_registry()["templates"][suite]["components"]["cumulus"] == "off"
    assert verdict["named_suite_label"] == physics_registry()["templates"][suite]["label"]


def test_the_wizard_writes_picked_schemes_on_every_size_it_tries_and_keeps_a_nests_cumulus_off(tmp_path):
    """`woof domain --physics-choices`: the mix is the file's, the fit prices it, and a nest stays convection-free.

    The write is `woof physics-catalog --into`'s, which used to put a changed cumulus scheme into every
    [[domain]] table, so a 12 km root's Grell-Freitas turned cumulus on at its 3 km nest too.
    """

    from woof.cli import build_parser
    from woof.domain_wizard import domain_main
    from woof.experiment import load_experiment

    choices = {"microphysics": "thompson-mp8", "cumulus": "grell-freitas"}
    assert pc.check({"choices": choices, "dx_km": 12, "source": "gfs"})["valid"]
    out = tmp_path / "mix.toml"
    args = build_parser().parse_args([
        "domain", "--point", "35.5,-97.5", "--source", "gfs", "--cycle", "2026-09-24T18", "--hours", "1",
        "--card", "16gb", "--ladder", "12-3", "--physics-choices", json.dumps(choices), "--out", str(out)])
    args.interactive = False
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert domain_main(args) == 0
    exp = load_experiment(out)
    assert [d.run.mp_physics for d in exp.domains] == [8, 8]
    assert [d.run.cu_physics for d in exp.domains] == [3, 0]
    assert "# PHYSICS MIX: cumulus grell-freitas, microphysics thompson-mp8 replace" in out.read_text(encoding="utf-8")
    assert "no suite is asserted" in printed.getvalue()
    refused = build_parser().parse_args([
        "domain", "--point", "35.5,-97.5", "--source", "gfs", "--cycle", "2026-09-24T18", "--hours", "1",
        "--card", "16gb", "--physics-choices", '{"pbl": "myj", "surface_layer": "revised-mm5"}',
        "--out", str(tmp_path / "no.toml")])
    refused.interactive = False
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match="sf_sfclay_physics=2"):
        domain_main(refused)
    assert not (tmp_path / "no.toml").exists()


def test_a_plan_with_picked_schemes_hands_them_to_the_wizard_and_asserts_no_suite(tmp_path):
    from woof.runplan import PlanError, _asserted_profile, build_plan, intent_arguments, manifest_physics

    base = pc.default_suite("gfs")
    intent = {"point": "35.5,-97.5", "source": "gfs", "cycle": "2026-09-24T18", "hours": 1, "card": "16gb",
              "root_dx_km": 3, "physics_profile": base, "physics_choices": THOMPSON_MYJ_ETA}
    argv = intent_arguments(intent, out=tmp_path / "x.toml")
    assert json.loads(argv[argv.index("--physics-choices") + 1]) == THOMPSON_MYJ_ETA
    raw = {"schema": "gpuwm.run-plan.v1", "name": "p", "route": "prepared", "config": {"intent": intent},
           "output_root": str(tmp_path / "out")}
    plan = build_plan(raw, source="test", base_dir=tmp_path, sha256="0" * 64)
    assert _asserted_profile(plan, config_path=tmp_path / "unread.toml") is None
    recorded = manifest_physics(plan)
    assert recorded["choices"] == THOMPSON_MYJ_ETA and recorded["base_suite"] == base and recorded["suite"] is None
    assert recorded["components"]["pbl"] == "myj" and recorded["switches"]["bl_pbl_physics"] == 2
    # The preparer holds a config to a suite named in run_options switch for switch; the mix is written over it.
    with pytest.raises(PlanError, match="physics_choices"):
        build_plan({**raw, "run_options": {"physics_profile": base}}, source="test", base_dir=tmp_path,
                   sha256="0" * 64)


def _domain(argv):
    """`woof domain ARGV`, run in this process past the CLI's install-identity door."""

    from woof.cli import build_parser
    from woof.domain_wizard import domain_main

    args = build_parser().parse_args(["domain", *argv])
    args.interactive = False
    return domain_main(args)


def test_a_mix_from_hrrr_is_written_as_the_route_runs_it_and_passes_its_round_trip(tmp_path):
    """The HRRR route runs its namelists, which have no key for moist_cq.

    The importer and catalog share the implicit-switch authority. An
    unmatched moist suite retains its pressure correction through the
    emitted configuration and namelist round trip.
    """

    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import route_input_paths, verify_round_trip

    # The mix is made over the suite HRRR bound by default until 4193eb0da
    # made it the GSD MYNN 4.1 suite.  Its scalar_pblmix=1 was coupled to
    # mp_physics=28 only, so the same mix was refused over that default
    # until U-E4; now the scheme's own scalars are diffused and the fork
    # Thompson generation goes back to its default with mp_physics 28.
    over_default = pc.check({"choices": THOMPSON_MYJ_ETA, "dx_km": 3, "source": "hrrr"})
    assert over_default["valid"], over_default.get("refusal")
    assert over_default["changed_from_suite"]["thompson_version"] == "wrf_461"
    base = "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1"
    verdict = pc.check({"suite": base, "choices": THOMPSON_MYJ_ETA, "dx_km": 3, "source": "hrrr"})
    assert verdict["valid"] and verdict["named_suite"] is None
    assert verdict["changed_from_suite"].get("moist_cq", True) is True
    # A route that reads the configuration itself runs the row's value.
    assert pc.check({"choices": THOMPSON_MYJ_ETA, "dx_km": 3, "source": "gfs"})["changed_from_suite"].get(
        "moist_cq", True) is True
    out = tmp_path / "mix.toml"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        rc = _domain(["--source", "hrrr", "--cycle", "2026-07-29T18", "--hours", "1", "--root-dx", "3",
                      "--chain", "3", "--card", "32gb", "--point", "38.0,-98.0",
                      "--physics-profile", base,
                      "--physics-choices", json.dumps(THOMPSON_MYJ_ETA), "--out", str(out)])
    assert rc == 0, printed.getvalue()
    exp = load_experiment(out)
    root = exp.root.run
    assert (root.mp_physics, root.bl_pbl_physics, root.sf_sfclay_physics, root.moist_cq) == (8, 2, 2, True)
    paths = route_input_paths(out)
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
    # The header names what the root runs, not the suite the mix replaced.
    physics = next(line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("# PHYSICS: "))
    assert physics.startswith("# PHYSICS: schemes picked over ") and "bl_pbl_physics 2" in physics, physics
    assert "physics: schemes picked over " in printed.getvalue()


def test_a_picked_cumulus_scheme_is_kept_at_a_storm_resolving_root_and_the_file_says_so(tmp_path):
    from woof.experiment import load_experiment

    choices = {"microphysics": "thompson-mp8", "cumulus": "grell-freitas"}
    verdict = pc.check({"choices": choices, "dx_km": 3, "source": "gfs"})
    assert verdict["valid"] and verdict["resolved"]["cumulus"] == "grell-freitas"
    out = tmp_path / "gf.toml"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        rc = _domain(["--source", "gfs", "--cycle", "2026-09-24T18", "--hours", "1", "--root-dx", "3",
                      "--chain", "3", "--card", "16gb", "--point", "35.5,-97.5",
                      "--physics-choices", json.dumps(choices), "--out", str(out)])
    assert rc == 0, printed.getvalue()
    text = out.read_text(encoding="utf-8")
    assert load_experiment(out).root.run.cu_physics == 3
    # The grid did not retire a scheme the user picked, so neither the file nor the screen says it did.
    assert "CUMULUS OFF" not in text and "CUMULUS OFF" not in printed.getvalue()
    assert "Grell-Freitas cumulus" in next(line for line in text.splitlines() if line.startswith("# PHYSICS: "))
