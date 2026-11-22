"""WRF v4.8.0 generations of NTU (mp 56) and UDM (mp 27): the breakages
these tests prevent are (1) the v4.8.0 selection compiling the v4.7.1
arithmetic because a source anchor moved, (2) the v4.7.1 selection no longer
compiling today's frozen source, and (3) a checkpoint continuing across a
generation change.  CPU only; the word replays live in tools/*_oracle."""
from __future__ import annotations

import pytest

from woof import mp_generations as g
from woof.config import ConfigurationRefused, RunConfig


def _cfg(**kw) -> RunConfig:
    return RunConfig(nx=6, ny=4, nz=5, dx=2000, dy=2000, ztop=8000, dt=1,
                     run_seconds=0, moist=True, **kw)


def _expected_by_wrf(monkeypatch):
    from pathlib import Path
    tool = Path(__file__).resolve().parents[1] / "tools" / "ntu_wrf471_oracle"
    monkeypatch.syspath_prepend(str(tool))
    import generate
    return generate.EXPECTED_BY_WRF


def _transcription(source: str) -> list[str]:
    """The f2cu.py block inside the composed ntu.cu, as lines."""
    begin = "// ---- BEGIN transcription (f2cu.py) ----\n"
    rest = source.split(begin, 1)[1]
    return rest.split("// ---- END transcription ----\n", 1)[0].splitlines(True)


#: f2cu.py's v4.7.1 -> v4.8.0 transcription diff (generate.py --wrf 4.8.0,
#: pinned in EXPECTED_BY_WRF) is one statement moved six lines down, after
#: the riming density clamp.  The shipped block differs from f2cu's output
#: only in A146's __fdiv_rn spelling of literal divisions, so the same move
#: applied to it must give the v4.8.0 composition.
F2CU_480_RATIO_DISPLACEMENT = 6


def test_ntu_480_moves_the_statement_f2cu_moves(monkeypatch):
    from woof.core.ntu import module_source
    expected = _expected_by_wrf(monkeypatch)
    assert set(expected) == {"4.7.1", "4.8.0"} and all(expected.values())
    old = _transcription(module_source(64, version="wrf_471"))
    new = _transcription(module_source(64, version="wrf_480"))
    assert old.count(g._NTU_RATIO) == 1
    at = old.index(g._NTU_RATIO)
    moved = old[:at] + old[at + 1:]
    moved.insert(at + F2CU_480_RATIO_DISPLACEMENT, g._NTU_RATIO)
    assert moved == new
    # Outside the moved statement the two compositions are the same text.
    whole_old = module_source(64, version="wrf_471")
    whole_new = module_source(64, version="wrf_480")
    assert whole_old.replace(g._NTU_RATIO, "") == whole_new.replace(g._NTU_RATIO, "")


def test_ntu_471_is_todays_source_byte_for_byte():
    from woof.core.kernels import module_source_int_defines
    base = module_source_int_defines("ntu", (("NTU_KMAX", 64),))
    assert g.ntu_source(base, "wrf_471") is base


def test_udm_480_adds_hail_deposition_and_zeroed_fall_densities():
    from woof.core.udm import module_source
    old = module_source(64, "wrf_471")
    new = module_source(128, "wrf_480")
    assert "-phdep[0+(k-(kts))*(1)]);" not in old
    assert new.count("-phdep[0+(k-(kts))*(1)]);\n") == 1
    for name in ("denqh", "denqg"):
        zero = f"{name}[0+(udmslice0-(kts))*(1)] = 0.f;"
        assert zero not in old and new.count(zero) == 1
    assert "#define UDM_COLUMN_KMAX 128" in new


def test_udm_471_is_the_loaders_source():
    from woof.core import kernels
    from woof.core.udm import module_source
    assert module_source(64, "wrf_471") == kernels.module_source("udm")
    assert module_source(256, "wrf_471") == kernels.module_source_int_defines(
        "udm", (("UDM_COLUMN_KMAX", 256),))


@pytest.mark.parametrize("compose", [
    lambda text: g.ntu_source(text, "wrf_480"),
    lambda text: g.udm_header(text, "wrf_480")])
def test_a_moved_anchor_refuses_rather_than_compiles_old_arithmetic(compose):
    with pytest.raises(RuntimeError, match="anchor"):
        compose("// no WRF statements here\n")


def test_unknown_generation_is_refused_by_name():
    with pytest.raises(ValueError, match="ntu_version"):
        g.ntu_source("", "wrf_461")
    with pytest.raises(ConfigurationRefused, match="udm_version"):
        from woof.config import validate_run_config
        validate_run_config(_cfg(mp_physics=27, udm_version="wrf_461"))


@pytest.mark.parametrize("mp_physics,selector", [(56, "ntu_version"), (27, "udm_version")])
def test_selector_binds_only_its_scheme_at_the_new_generation(mp_physics, selector):
    values = {"mp_physics": mp_physics, "ntu_version": "wrf_480", "udm_version": "wrf_480"}
    g.drop_inactive(values)
    assert values == {"mp_physics": mp_physics, selector: "wrf_480"}
    values = {"mp_physics": mp_physics, selector: "wrf_471"}
    g.drop_inactive(values)
    assert values == {"mp_physics": mp_physics}


def test_restart_identity_follows_the_generation():
    from woof.checkpoint_identity import MICROPHYSICS_ALGORITHM_IDENTITIES as table
    for mp_physics, selector in ((56, "ntu_version"), (27, "udm_version")):
        old = _cfg(mp_physics=mp_physics, **{selector: "wrf_471"})
        new = _cfg(mp_physics=mp_physics)
        assert g.microphysics_algorithm_identity(old, table[mp_physics]) == table[mp_physics]
        assert "wrf480" in g.microphysics_algorithm_identity(
            new, table[mp_physics]).replace("v4.8.0", "wrf480")
    other = _cfg(mp_physics=8)
    assert g.microphysics_algorithm_identity(other, table[8]) == table[8]


def test_a_wrf_47_namelist_imports_the_previous_generation():
    from woof.namelist_import import wrf_release_generations
    selected = wrf_release_generations("4.7.1")
    assert selected["ntu_version"] == selected["udm_version"] == "wrf_471"
    assert "ntu_version" not in wrf_release_generations("4.8.0")
