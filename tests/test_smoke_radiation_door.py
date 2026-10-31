"""The explicit radiation door preserves transport and omitted-flag behavior."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.smoke_radiation import apply_smoke_radiation, at_flag


def test_document_choice_uses_shared_scope_without_changing_transport():
    document = {"shared": {"chem_sets": "smoke", "chem_sources": "rave-3km",
                           "simple_dir_fdb": 1, "aer_opt": 3,
                           "aerosol_mp_coupling": "none", "smoke_feedback": ""},
                "domain": [{"grid_id": 1}, {"grid_id": 2, "radt": 15.0}]}
    original = deepcopy(document)
    assert apply_smoke_radiation(document, "on") is document
    row = document["shared"]
    assert (row["aer_ra_feedback"], row["simple_dir_fdb"], row["aer_opt"], row["aer_op_opt"]) == (1, 0, 0, 1)
    assert document["domain"] == original["domain"]
    for key in ("chem_sets", "chem_sources", "aerosol_mp_coupling", "smoke_feedback"):
        assert document["shared"][key] == original["shared"][key]
    apply_smoke_radiation(document, "off")
    assert row["aer_ra_feedback"] == row["simple_dir_fdb"] == 0
    assert document["shared"]["chem_sets"] == "smoke"


@pytest.mark.parametrize("document", [
    {}, {"shared": {"chem_sets": ""}},
    {"shared": {"chem_sets": "smoke"}, "domain": [{"chem_sets": ""}]},
])
def test_no_optics_refuses_without_partial_mutation(document):
    original = deepcopy(document)
    with pytest.raises(ValueError, match="smoke-radiation-no-optics"):
        apply_smoke_radiation(document, "on")
    assert document == original


def test_off_retains_background_aerosol_and_accepts_no_chemistry():
    doc = {"shared": {"aer_opt": 3, "simple_dir_fdb": 1}}
    apply_smoke_radiation(doc, "off")
    assert doc["shared"] == {"aer_opt": 3, "simple_dir_fdb": 0, "aer_ra_feedback": 0}


def test_omitted_flag_is_exact_noop():
    config, payload = object(), object()
    assert at_flag(SimpleNamespace(), config, payload) == (config, payload)


def test_prepared_flag_names_identity_breakage(tmp_path):
    with pytest.raises(ValueError, match="smoke-radiation-prepared-identity"):
        at_flag(SimpleNamespace(smoke_radiation="on", prepared_root=tmp_path),
                tmp_path / "input.toml", {})


@pytest.mark.parametrize("mode", ["on", "off"])
def test_go_door_reaches_resolved_config_and_namelist(tmp_path, monkeypatch, mode):
    from woof import go_cli
    from woof.cli import main
    from woof.core import preflight
    from woof.experiment import load_experiment
    recipe = Path(__file__).resolve().parents[1] / "woof/data/recipes/hrrr_v4_gsd41.toml"
    original = recipe.read_bytes()
    captured = {}

    def launch(args, *, config, **kwargs):
        captured.update(path=config, experiment=load_experiment(config))
        return 0

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: None)
    monkeypatch.setattr(go_cli, "_registered_launch", launch)
    assert main(["go", str(recipe), "--smoke", "--smoke-radiation", mode,
                 "--outdir", str(tmp_path / "run"),
                 "--wps-namelist", str(tmp_path / "input.namelist.wps")]) == 0
    cfg = captured["experiment"].root.run
    assert cfg.chem_sets == "smoke"
    assert cfg.aer_ra_feedback == int(mode == "on")
    assert cfg.simple_dir_fdb == 0
    if mode == "on":
        assert (cfg.aer_opt, cfg.aer_op_opt) == (0, 1)
    assert recipe.read_bytes() == original
    from woof.hrrr_route_inputs import render_namelist_input
    import re
    namelist = render_namelist_input(captured["experiment"])
    assert re.search(r"\baer_ra_feedback\s*=\s*" + str(int(mode == "on")) + r"\b", namelist)
    assert re.search(r"\bsimple_dir_fdb\s*=\s*0\b", namelist)


def test_explicit_on_cannot_fall_back_to_chemistry_off(tmp_path, monkeypatch):
    from woof import go_cli, smoke_execution
    from woof.cli import main
    from woof.core import preflight
    recipe = Path(__file__).resolve().parents[1] / "woof/data/recipes/hrrr_v4_gsd41.toml"
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: None)
    monkeypatch.setattr(smoke_execution, "capacity_reason", lambda *a, **kw: "test resident envelope exceeds memory")
    monkeypatch.setattr(go_cli, "_registered_launch", lambda *a, **kw: pytest.fail("silently launched"))
    assert main(["go", str(recipe), "--smoke", "--smoke-radiation", "on",
                 "--outdir", str(tmp_path / "run"),
                 "--wps-namelist", str(tmp_path / "input.namelist.wps")]) != 0
