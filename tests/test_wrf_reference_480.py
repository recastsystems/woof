"""WRF v4.8.0 is WOOF-ArWen's referee; 4.7.1 stays selectable (woof.wrf_reference).

The breakage this prevents: the importer read only WRF 4.7.1's Registry, so
a namelist written for wrf.exe 4.8.0 was refused key by key ("unmapped
key") for keys 4.8.0 itself declares at their defaults (the Noah-MP runoff
split, the MYNN/TEMPO/KIM flags, &cordex), and there was no one switch to
pin a run to the previous release.  These tests hold:

* the packaged 4.8.0 table is WRF 4.8.0's Registry (its delta from 4.7.1
  is exactly the one upstream made), and every key 4.8.0 added has an
  owner here, so no 4.8.0 key reaches a user unclassified;
* WRF's own 4.8.0 namelists read the same with every new key written at
  its default as without (the round trip), and a non-default value is
  refused by name;
* the 4.7.1 vocabulary behaves as before (option-off identity);
* ``wrf_reference`` in a run TOML is exactly the per-scheme keys it names.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tomllib
import warnings

import pytest

from woof.fortran_namelist import parse_namelist
from woof.namelist_contract import _base_pair, _registry_column, _registry_value
from woof.namelist_import import (WRF_PHYSICS_SUITES, import_parsed_namelists,
                                   namelist_refusals)
from woof.wrf_namelist_registry import (REGISTRY_TABLES, wrf_namelist_keys,
                                         wrf_namelist_table)
from woof.wrf_reference import (CORDEX_GROUP, REFERENCE_STOCK_SELECTORS, NOAHMP_RUNOFF_SPLIT,
                                 WRF480_ONLY_KEYS, WRF480_PORTED_KEYS, WRF_REFERENCE_SOURCES,
                                 check_wrf_reference,
                                 registry_reference_for_wrf_version,
                                 resolve_wrf_reference,
                                 wrf_reference_receipt)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "wrf480_namelists"

#: Keys WRF 4.8.0 added that the importer reads outside WRF480_ONLY_KEYS.
READ_ELSEWHERE = {("physics", "bl_mynn_mixaerosols"),
                  ("physics", "bl_mynn_mixnumcon"),
                  ("physics", "windfarm_induction"),
                  *(("noah_mp", key) for key in NOAHMP_RUNOFF_SPLIT)}

WRF_471_GENERATIONS = {"solar_generation": "wrf_461",
                       "ysu_version": "wrf_461",
                       "shinhong_version": "wrf_461",
                       "urban_water_balance": "wrf_471",
                       "ntu_version": "wrf_471",
                       "udm_version": "wrf_471",
                       "cu_gf_version": "wrf_461",
                       "gwdo_version": "wrf_471",
                       "windfarm_fitch_version": "wrf_471"}


def _import(wps, inp, version):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return import_parsed_namelists(
            deepcopy(wps), deepcopy(inp), wps_path="namelist.wps",
            input_path="namelist.input", wrf_version=version,
            wrf_version_source=f"--wrf-version {version}")[0]


def _refusals(wps, inp, version):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return namelist_refusals(deepcopy(wps), deepcopy(inp),
                                 wrf_version=version)


def _with_480_defaults(inp):
    registry = wrf_namelist_keys("4.8.0")
    out = deepcopy(inp)
    count = max(int(out.get("domains", {}).get("max_dom", [1])[0]), 1)
    for section, key in WRF480_ONLY_KEYS:
        out.setdefault(section, {})[key] = _registry_column(
            registry[(section, key)], count)
    out[CORDEX_GROUP] = {"output_cordex": [0], "psl_diag": [3]}
    return out


# --------------------------------------------------------------- tables

def test_the_480_table_is_wrf_480s_registry():
    table = wrf_namelist_table("4.8.0")
    assert table["wrf"]["tag"] == WRF_REFERENCE_SOURCES["4.8.0"]["tag"]
    assert table["wrf"]["commit"] == WRF_REFERENCE_SOURCES["4.8.0"]["commit"]
    assert wrf_namelist_table("4.7.1")["wrf"]["tag"] == "v4.7.1"
    old, new = wrf_namelist_keys("4.7.1"), wrf_namelist_keys("4.8.0")
    assert set(old) - set(new) == {("noah_mp", "opt_run")}
    changed = {key for key in set(old) & set(new) if old[key] != new[key]}
    assert changed == {("physics", "bl_mynn_mixscalars")}
    assert (old[("physics", "bl_mynn_mixscalars")]["default"],
            new[("physics", "bl_mynn_mixscalars")]["default"]) == ("1", "0")
    assert len(new) == 1176 and len(old) == 1115


def test_every_key_480_added_has_an_owner():
    added = set(wrf_namelist_keys("4.8.0")) - set(wrf_namelist_keys("4.7.1"))
    owned = (set(WRF480_ONLY_KEYS) | set(WRF480_PORTED_KEYS) | READ_ELSEWHERE
             | {key for key in added if key[0] == CORDEX_GROUP})
    assert added - owned == set(), sorted(added - owned)
    assert set(WRF480_ONLY_KEYS) <= added


def test_the_release_chooses_the_vocabulary():
    for version, reference in (("4", "4.8.0"), ("4.8", "4.8.0"),
                               ("4.8.0", "4.8.0"), ("4.9.1", "4.8.0"),
                               ("4.7.1", "4.7.1"), ("4.6.1", "4.7.1"),
                               ("3", "4.7.1"), ("3.9", "4.7.1")):
        assert registry_reference_for_wrf_version(version) == reference
    assert check_wrf_reference("v4.8.0") == "4.8.0"
    with pytest.raises(ValueError, match="4.8.0"):
        check_wrf_reference("4.6.1")
    assert sorted(REGISTRY_TABLES) == ["4.7.1", "4.8.0"]


# --------------------------------------------------------- round trip

@pytest.mark.parametrize("path", sorted(FIXTURES.glob("namelist.input*")),
                         ids=lambda path: path.name)
def test_wrf_480s_own_namelists_read_the_same_with_every_new_key(path):
    """WRF 4.8.0's test/em_real namelists, unchanged and with every 4.8.0
    key written at its default: the same TOML, or the same refusals."""
    inp = parse_namelist(path)
    wps = _base_pair()[0]
    if int(inp["domains"].get("max_dom", [1])[0]) != 2:
        # The base WPS is two domains; read the root alone otherwise.
        inp = deepcopy(inp)
        for section in inp.values():
            for key, values in list(section.items()):
                if isinstance(values, list) and len(values) > 1:
                    section[key] = values[:1] if key != "eta_levels" else values
        inp["domains"]["max_dom"] = [1]
        wps = deepcopy(wps)
        wps["share"]["max_dom"] = [1]
        for key in ("start_date", "end_date"):
            wps["share"][key] = wps["share"][key][:1]
        for key, values in wps["geogrid"].items():
            wps["geogrid"][key] = values[:1]

    def outcome(document):
        problems = _refusals(wps, document, "4.8.0")
        if problems:
            return sorted({key for problem in problems
                           for key in getattr(problem, "keys", ())}), None
        return [], _import(wps, document, "4.8.0")

    assert outcome(_with_480_defaults(inp)) == outcome(inp)


def test_every_new_key_at_its_default_is_inert_and_otherwise_refused():
    wps, inp = _base_pair()
    base = _import(wps, inp, "4.8.0")
    assert _import(wps, _with_480_defaults(inp), "4.8.0") == base
    assert _import(wps, _with_480_defaults(inp), "4") == _import(wps, inp, "4")
    registry = wrf_namelist_keys("4.8.0")
    for (section, key), row in WRF480_ONLY_KEYS.items():
        value = _registry_value(registry[(section, key)])
        bad = (not value if isinstance(value, bool) else value + 7)
        document = deepcopy(inp)
        document.setdefault(section, {})[key] = [bad, bad]
        problems = _refusals(wps, document, "4.8.0")
        assert any((section, key) in getattr(p, "keys", ()) for p in problems), key
        assert any(row["owner"] in str(p) for p in problems), key


def test_cordex_is_inert_off_and_refused_on():
    wps, inp = _base_pair()
    document = deepcopy(inp)
    document[CORDEX_GROUP] = {"output_cordex": [1]}
    problems = _refusals(wps, document, "4.8.0")
    assert any((CORDEX_GROUP, "output_cordex") in p.keys for p in problems)


def test_the_noahmp_runoff_split_reads_as_opt_run():
    wps, inp = _base_pair()
    inp = deepcopy(inp)
    inp["physics"]["sf_surface_physics"] = [4, 4]
    base = _import(wps, inp, "4.8.0")
    same = deepcopy(inp)
    same["noah_mp"] = {"opt_runsrf": [3], "opt_runsub": [3]}
    assert _import(wps, same, "4.8.0") == base
    one = deepcopy(inp)
    one["noah_mp"] = {"opt_runsrf": [1], "opt_runsub": [1]}
    assert "opt_run = 1" in _import(wps, one, "4.8.0").splitlines()
    # Separate surface and subsurface runoff imports since the Noah-MP clean
    # program's split (opt_run carries the surface half, opt_runsub the other).
    split = deepcopy(inp)
    split["noah_mp"] = {"opt_runsrf": [1], "opt_runsub": [2]}
    lines = _import(wps, split, "4.8.0").splitlines()
    assert "opt_run = 1" in lines and "opt_runsub = 2" in lines
    # A pair the split kernel cannot run is refused by name (the Niu 2007
    # saturated fraction needs a groundwater subsurface half).
    bad = deepcopy(inp)
    bad["noah_mp"] = {"opt_runsrf": [1], "opt_runsub": [3]}
    with pytest.raises(Exception, match="opt_runsrf=1 with opt_runsub=3"):
        _import(wps, bad, "4.8.0")
    # Under the previous pin the keys stay what wrf.exe 4.7.1 calls them:
    # unknown, refused by name.
    problems = _refusals(wps, same, "4.7.1")
    assert any(("noah_mp", "opt_runsrf") in p.keys for p in problems)


def test_the_471_vocabulary_refuses_480_keys_as_before():
    wps, inp = _base_pair()
    for (section, key) in (list(WRF480_ONLY_KEYS)[:3]
                           + list(WRF480_PORTED_KEYS)[:3]):
        document = deepcopy(inp)
        document.setdefault(section, {})[key] = _registry_column(
            wrf_namelist_keys("4.8.0")[(section, key)], 2)
        problems = _refusals(wps, document, "4.7.1")
        assert any((section, key) in p.keys for p in problems), key
        assert any("not a WRF v4.7.1 namelist key" in str(p)
                   for p in problems), key


def test_the_tropical_suite_follows_the_release():
    wps, inp = _base_pair()
    inp = deepcopy(inp)
    inp["physics"].update(physics_suite=["tropical"],
                          sf_sfclay_physics=[-1, -1])
    assert WRF_PHYSICS_SUITES["tropical"]["sf_sfclay_physics"] == 91

    def sfclay(version):
        shared = tomllib.loads(_import(wps, inp, version))["shared"]
        return shared["sf_sfclay_physics"]
    assert sfclay("4.8.0") == 1 and sfclay("4") == 1
    assert sfclay("4.7.1") == 91


# ---------------------------------------------------------- run doors

def test_resolve_writes_the_releases_keys_and_an_explicit_key_wins():
    table = {"wrf_reference": "4.7.1"}
    assert resolve_wrf_reference(table, where="t") == "4.7.1"
    assert table == {**WRF_471_GENERATIONS, **REFERENCE_STOCK_SELECTORS}
    table = {"wrf_reference": "4.7.1", "ysu_version": "wrf_480"}
    resolve_wrf_reference(table, where="t")
    assert table["ysu_version"] == "wrf_480"
    table = {"wrf_reference": "4.8.0"}
    resolve_wrf_reference(table, where="t")
    assert table == REFERENCE_STOCK_SELECTORS
    untouched = {"ysu_version": "wrf_461"}
    assert resolve_wrf_reference(untouched, where="t") is None
    assert untouched == {"ysu_version": "wrf_461"}
    with pytest.raises(ValueError, match="wrf_reference in t"):
        resolve_wrf_reference({"wrf_reference": "4.6"}, where="t")


def _experiment(shared_extra):
    from woof.experiment import build_experiment
    wps, inp = _base_pair()
    raw = tomllib.loads(_import(wps, inp, "4"))
    raw["shared"].update(shared_extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_experiment(raw, source="test")


def test_the_experiment_door_pins_exactly_the_per_scheme_keys():
    pinned = _experiment({"wrf_reference": "4.7.1"})
    explicit = _experiment({**WRF_471_GENERATIONS, **REFERENCE_STOCK_SELECTORS})
    assert [d.run for d in pinned.domains] == [d.run for d in explicit.domains]
    assert wrf_reference_receipt(pinned.domains[0].run)["wrf_reference"] == "4.7.1"
    default = _experiment({})
    # 4.8.0 is every engine generation default plus the stock-WRF RUC
    # SOILPROP form (REFERENCE_STOCK_SELECTORS), which the engine default
    # deliberately is not.
    assert [d.run for d in _experiment({"wrf_reference": "4.8.0"}).domains] \
        == [d.run for d in _experiment(dict(REFERENCE_STOCK_SELECTORS)).domains]
    assert wrf_reference_receipt(default.domains[0].run)["wrf_reference"] == "4.8.0"


def test_the_legacy_door_pins_exactly_the_per_scheme_keys(tmp_path):
    from woof.config import load_config
    source = (ROOT / "configs" / "real74_d01.toml").read_text(encoding="utf-8")
    pinned = tmp_path / "pinned.toml"
    pinned.write_text(source.replace(
        "[run]\n", '[run]\nwrf_reference = "4.7.1"\n', 1), encoding="utf-8")
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(source.replace("[run]\n", "[run]\n" + "".join(
        f'{key} = "{value}"\n' for key, value in
        {**WRF_471_GENERATIONS, **REFERENCE_STOCK_SELECTORS}.items()),
        1), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert load_config(pinned) == load_config(explicit)
        assert load_config(pinned) != load_config(ROOT / "configs" / "real74_d01.toml")


def test_the_packaged_tables_regenerate_from_their_own_rows():
    """The JSON is the tool's output shape (tools/wrf_namelist_registry.py)."""
    for reference, path in REGISTRY_TABLES.items():
        table = json.loads(path.read_text(encoding="utf-8"))
        assert table["schema"] == "gpuwm.wrf-namelist-registry.v1"
        assert path.read_text(encoding="utf-8") == json.dumps(table, indent=1) + "\n"
        rows = [(r["section"], r["key"]) for r in table["keys"]]
        assert rows == sorted(rows)
