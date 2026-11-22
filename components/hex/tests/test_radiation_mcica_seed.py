"""config_radt_mcica_seed: the cloud-sample seed rule of RRTMG's McICA generator (woof.hex.mpasphys.mcica_seed_rule)
and its adjoint-door side (woof.hex.ad_radiation_seed)."""
import json

import numpy as np
import pytest

from woof.hex import ad_radiation_seed as radseed
from woof.hex.mpasphys import mcica_seed_rule as rule


def test_the_default_is_mpas_rule_and_launches_nothing():
    assert rule.DEFAULT == rule.PRESSURE == rule.mode({})
    assert rule.mode({rule.OPTION: "pressure"}) == rule.PRESSURE
    assert rule.held({}, lambda: pytest.fail("the pressure rule reads no mesh"), 7) is None
    assert rule.held({rule.OPTION: "pressure"}, lambda: pytest.fail("no mesh"), 7) is None


def test_an_unknown_rule_is_refused_by_name():
    with pytest.raises(ValueError, match=rule.OPTION):
        rule.mode({rule.OPTION: "cel"})


def test_every_suite_table_carries_the_word_with_mpas_rule_as_default():
    from woof.hex.physics_suites import load_table
    table = load_table()
    for key in ("mpasphys_v842", "mpasphys_rrfs", "mpasphys_rrfs_44c0849"):
        assert table[key]["registry_defaults"][rule.OPTION] == {"type": "character", "value": "pressure"}
    for key in ("mpasphys_rrfs", "mpasphys_rrfs_44c0849"):
        assert rule.OPTION in table[key]["registry_groups"]["physics"]


def test_cell_words_depend_on_the_cell_step_and_band_only():
    a = rule.host_words([1, 2, 2890], 30, rule.LONGWAVE)
    assert (a == rule.host_words([1, 2, 2890], 30, rule.LONGWAVE)).all()
    assert (a[1] == rule.host_words([2], 30, rule.LONGWAVE)[0]).all()          # the batch does not matter
    assert not (a == rule.host_words([1, 2, 2890], 31, rule.LONGWAVE)).any()   # the step does
    assert not (a == rule.host_words([1, 2, 2890], 30, rule.SHORTWAVE)).any()  # the band does
    assert (a != 0).all()


def test_forecast_words_name_the_rule():
    assert radseed.forecast_seed(["forecast", "--mesh", "x"]) is None
    assert radseed.forecast_seed(["--physics", "config_radt_mcica_seed=cell"]) == "cell"
    assert radseed.forecast_seed(["--physics=config_radt_mcica_seed='pressure'",
                                  "--physics", "config_radt_mcica_seed=cell"]) == "cell"


def _package(tmp_path, kernels, words=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "plan.json").write_text(json.dumps({"events": [{"function": k} for k in kernels]}))
    (tmp_path / "record-receipt.json").write_text(json.dumps({"forecast_args": ["forecast", *words]}))
    return tmp_path


def test_a_plan_records_its_rule_and_refuses_the_other_by_name(tmp_path, capsys):
    from woof.hex.ad_doors import AdDoorError
    pressure = _package(tmp_path / "p", ["mpas_lw_mcica_seeds", "mpas_sw_mcica_seeds"])
    cell = _package(tmp_path / "c", ["mpas_lw_mcica_seeds", rule.KERNEL],
                    ["--physics", "config_radt_mcica_seed=cell"])
    none = _package(tmp_path / "n", ["acoustic_step"])
    assert radseed.plan_seed(pressure)["rule"] == "pressure"
    assert radseed.plan_seed(cell)["rule"] == "cell"
    assert radseed.plan_seed(none)["rule"] is None
    assert radseed.refuse_plan_seed(pressure, None, "tangent") == "pressure"
    assert radseed.refuse_plan_seed(cell, "cell", "tangent") == "cell"
    assert "cell rule" in capsys.readouterr().out
    with pytest.raises(AdDoorError, match="config_radt_mcica_seed = 'pressure'"):
        radseed.refuse_plan_seed(pressure, "cell", "adjoint-test")
    with pytest.raises(AdDoorError, match="config_radt_mcica_seed = 'cell'"):
        radseed.refuse_plan_seed(cell, "pressure", "sensitivity")
    assert radseed.refuse_plan_seed(none, "cell", "tangent") is None
    broken = _package(tmp_path / "b", ["mpas_lw_mcica_seeds"], ["--physics", "config_radt_mcica_seed=cell"])
    with pytest.raises(AdDoorError, match="not one recording"):
        radseed.plan_seed(broken)


def test_a_recording_from_init_takes_the_forecast_word():
    class A:
        radiation_seed = None
    assert radseed.record_words(A()) == []
    A.radiation_seed = "pressure"
    assert radseed.record_words(A()) == []
    A.radiation_seed = "cell"
    assert radseed.record_words(A()) == ["--", "--physics", "config_radt_mcica_seed=cell"]


def test_the_runner_treats_the_launch_as_inactive():
    from pathlib import Path
    rules = Path(__file__).resolve().parents[1] / "rust" / "hexad" / "src" / "runner" / "wrules.rs"
    if not rules.is_file():
        pytest.skip("source tree only")
    assert f'"{rule.KERNEL}"' in rules.read_text()


@pytest.mark.gpu
def test_the_kernel_writes_the_host_words():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no card")
    ids = np.array([5, 1, 40962, 2890], dtype=np.int32)
    seeds = cp.zeros((4, 4), dtype=cp.uint32)
    rule.Held(cp.asarray(ids), 120).write(seeds, rule.SHORTWAVE)
    assert (seeds.get() == rule.host_words(ids, 120, rule.SHORTWAVE)).all()
    part = cp.zeros((2, 4), dtype=cp.uint32)
    rule.Held(cp.asarray(ids), 120).take(cp.asarray([3, 1])).write(part, rule.SHORTWAVE)
    assert (part.get() == rule.host_words(ids[[3, 1]], 120, rule.SHORTWAVE)).all()
