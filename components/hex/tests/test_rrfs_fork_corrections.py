"""Correct mode's fork scheme corrections (woof.hex.mpasphys.rrfs.fork_corrections): what each one changes, and that
exact mode composes the pinned sources byte for byte.  Processors only; the forecast proofs are run records."""
import pytest

from woof.hex.mpasphys.rrfs import fork_corrections as fc
from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
from woof.hex.mpasphys.rrfs import tempo_phase_v303 as phase
from woof.hex.mpasphys.rrfs import bl_mynnedmf_c6d08709 as mynn6, bl_mynnedmf_c942418 as mynn9
from woof.hex.mpasphys.rrfs import sf_mynnsfc_v600 as sfc
from woof.hex.mpasphys.rrfs.driver_rrfs import (Backend, SlotRegistry, resolve_namelist, route_namelist,
                                               specification)

ALL_TEMPO = ("TP1", "TP2", "TP3", "TP4")
ALL5 = ALL_TEMPO + ("TP5",)
DM = "// device maths stand-in\n"


def deck(**options):
    def value(v):
        return '.true.' if v is True else '.false.' if v is False else repr(v)
    return '&physics\n' + '\n'.join(f'{k}={value(v)}' for k, v in
                                    {'config_physics_suite': 'hrrrv5', **options}.items()) + '\n/'


def test_exact_sources_are_the_pinned_objects():
    # Breakage prevented: a correction leaking into exact mode, which must stay bit for bit with native.
    assert tempo.corrected_sources(()) == (tempo._COLUMN, tempo._DRIVER, tempo._WARP)
    assert tempo.corrected_sources(("MY1",)) == (tempo._COLUMN, tempo._DRIVER, tempo._WARP)
    assert tempo._column_source("strict") is tempo._COLUMN
    for mod in (mynn6, mynn9):
        assert mod.module_cuda() is mod.MODULE_CUDA and mod.module_cuda(ALL_TEMPO) is mod.MODULE_CUDA


def test_each_tempo_correction_changes_only_its_statement():
    column, driver, warp = tempo.corrected_sources(ALL_TEMPO)
    assert "if (fabs(ssatw[k]) < PS(eps)) continue;" in column
    assert "if (fabs(ssatw[k]) < PS(eps)) return;" not in column and "kstop" not in column.split(
        "inline void cloud_condensation(")[1].split("TEMPO_HDN inline void rain_evaporation(")[0]
    assert "T.pbg_rci[k] = PS(meters3_to_liters) * T.prg_rci[k] / PS(rho_i);" in column
    assert "T.pbg_scw[k] = T.pbg_scw[k] * ratio;" in column and "T.pbg_gcw[k] = T.pbg_gcw[k] * ratio;" in column
    assert "T.pbg_scw[k] = T.pbg_scw[k] * ratio;" not in tempo._COLUMN
    assert "first && c==0" not in driver and "first && sub==0 && c==0" not in warp
    one = {f: tempo.corrected_sources((f,)) for f in ALL_TEMPO}
    assert one["TP1"][1:] == one["TP2"][1:] == (tempo._DRIVER, tempo._WARP)
    assert one["TP3"][0] is tempo._COLUMN
    assert len(one["TP1"][0]) < len(tempo._COLUMN) and len(one["TP2"][0]) > len(tempo._COLUMN)


def test_corrected_phase_and_fast_sources_compose():
    # The fast transforms and the phase split must still find every anchor on the corrected column.
    for corr in ((), ALL_TEMPO, ("TP1",)):
        for numerics in ("strict", "fast"):
            assert tempo.cuda_source(DM, max_levels=59, numerics=numerics, corrections=corr)
            assert tempo.warp_source(DM, max_levels=59, numerics=numerics, corrections=corr)
        assert phase.phase_source(DM, 59, "", corr) and phase.tail_source(DM, 59, "", corr)
    assert "first && sub==0 && c==0" not in phase.tail_source(DM, 59, "", ALL_TEMPO)
    assert "first && sub==0 && c==0" in phase.tail_source(DM, 59, "")


def test_mynn_dew_limiter_is_upstreams():
    for mod in (mynn6, mynn9):
        source = mod.module_cuda(("MY1",))
        assert mod._MY1[0] not in source and source.count(mod._MY1[1]) == 1
        assert "r6_sub(1e-8f, r6_mul(0.9f, sqv(kts)))" in mod._MY1[1]
        assert "r6_mul(r6_mul(dtz(kts), rhosfc), rhoinv(kts))" in mod._MY1[1]
        both = mod.module_cuda(("MY1", "MY2"))
        assert mod._MY2[0] not in both and both.count(mod._MY2[1]) == 1 and both.count(mod._MY1[1]) == 1
        assert "r6_mul(r6_mul(r_d, tk(kts)), r6_add(one, r6_mul(p608, qv(kts))))" in mod._MY2[1]


def test_mynn_reworks_are_upstreams():
    # MY3 (e447691) and MY4 (87af8c6) on both pinned MYNN-EDMF texts: each pinned line goes, its corrected line
    # comes once, and ids of other schemes leave the text alone.
    for mod in (mynn6, mynn9):
        source = mod.module_cuda(("MY3", "MY4"))
        assert mod._MY3[0] not in source and source.count(mod._MY3[1]) == 1
        assert "sqc2(k)" in mod._MY3[1] and "sqi2(k)" in mod._MY3[1]
        assert len(mod._MY4) == 3
        for old, new in mod._MY4:
            assert old in mod.MODULE_CUDA and old not in source and source.count(new) == 1
            assert "r6_sub(1.f, sqv(k))" in new
        assert mod.module_cuda(("MY1", "MY2", "MY3", "MY4")).count(mod._MY1[1]) == 1
        assert mod.module_cuda(("SF1", "TP4")) is mod.MODULE_CUDA
        five = mod.module_cuda(("MY5",))
        for old, new in mod._MY5:
            assert old not in five and five.count(new) == 1
        assert "k = ((kthv) + (1));" in mod._MY5[1][0] and "mynn_max((int)(((kthv) + (1)))" in mod._MY5[1][1]


PINS = "\n&physics_mp_tempo config_tempo_version='v3.0.3' /"
K44 = "mpasphys_rrfs_44c0849"
SIX = {"MY1": "rrfs.bl_mynnedmf_v611", "MY2": "rrfs.bl_mynnedmf_v611", "MY3": "rrfs.bl_mynnedmf_v611",
       "MY4": "rrfs.bl_mynnedmf_v611", "MY5": "rrfs.bl_mynnedmf_v611",
       "SF1": "rrfs.sf_mynnsfc_v610",
       "TP1": "rrfs.tempo_v325", "TP2": "rrfs.tempo_v325", "TP3": "rrfs.tempo_v325"}


def test_exact_mode_applies_nothing_and_correct_mode_applies_every_fix_the_running_source_lacks():
    exact = route_namelist(deck(), physics_mode="exact")
    assert exact.mode == "exact" and not exact.corrections and not exact.gaps and not exact.fork_carried()
    # The older pin set has no MYNN version table: correct mode runs its pinned MYNN, and TEMPO v3.2.5.
    correct = route_namelist(deck(), physics_mode="correct")
    assert str(correct.backends["microphysics"]) == "hex-cuda" and str(correct.backends["pbl"]) == "hex-cuda"
    assert correct.selectors["microphysics"] == "mp_tempo" and correct.selectors["pbl"] == "bl_mynnedmf"
    assert correct.options["config_tempo_version"] == "v3.2.5"
    assert correct.corrections == frozenset({"MY1", "MY2", "MY3", "MY4", "MY5", "SF1", "TP4", "TP5", "TP6"})
    record = correct.receipt()["correct_mode"]
    assert record["applied"] == ["MY1", "MY2", "MY3", "MY4", "MY5", "SF1", "TP4", "TP5", "TP6"]
    assert record["in_scheme_versions"] == {k: v for k, v in SIX.items() if k.startswith("TP")}
    assert record["gaps"][0]["finding"] == specification(correct)["correct_mode_inventory"]["finding"]
    warning = correct.gap_warning()
    assert "applied fork corrections: MY1, MY2, MY3, MY4, MY5, SF1, TP4, TP5, TP6" in warning
    assert "already in the running scheme versions: TP1 (rrfs.tempo_v325), TP2 (rrfs.tempo_v325)" in warning
    # A correct-mode run that names the TEMPO pin gets the pin with all four TEMPO fixes.
    pinned = route_namelist(deck() + PINS, physics_mode="correct")
    assert pinned.options["config_tempo_version"] == "v3.0.3"
    assert pinned.corrections == frozenset({"MY1", "MY2", "MY3", "MY4", "MY5", "SF1", "TP1", "TP2", "TP3", "TP4", "TP5", "TP6"})
    assert not pinned.fork_carried()


def test_noaa_pin_set_default_carries_six_in_its_scheme_versions_and_applies_tp4_tp5_and_tp6():
    # Breakage prevented: a receipt naming MY1 applied while MYNN-EDMF v6.1.1 (whose own text has the fix, and
    # which declares no correction) runs the slot: SlotRegistry.require would refuse every default run.
    correct = route_namelist(deck(), physics_mode="correct", table_key=K44)
    assert correct.mynn_version == "mynn-v6.1" and correct.options["config_tempo_version"] == "v3.2.5"
    assert correct.corrections == frozenset({"TP4", "TP5", "TP6"})
    assert dict(correct.fork_carried()) == SIX
    assert correct.receipt()["correct_mode"]["in_scheme_versions"] == SIX
    # The surface-layer half alone: MYNN-EDMF stays the pin, so MY1 to MY5 are applied there.
    half = route_namelist(deck() + "\n&hex_mpasphys config_mynn_version='mynn-sfc-v6.1.0' /",
                          physics_mode="correct", table_key=K44)
    assert half.corrections == frozenset({"MY1", "MY2", "MY3", "MY4", "MY5", "TP4", "TP5", "TP6"})
    # Naming every pin in correct mode: every row is applied as a text pair.
    pins = route_namelist(deck() + PINS + "\n&hex_mpasphys config_mynn_version='pinned' /",
                          physics_mode="correct", table_key=K44)
    assert pins.corrections == frozenset(fc.FORK_CORRECTIONS) and not pins.fork_carried()
    exact = route_namelist(deck(), physics_mode="exact", table_key=K44)
    assert not exact.corrections and exact.mynn_version == "pinned" and exact.options["config_tempo_version"] == "v3.0.3"


def test_carried_by_names_modules_whose_text_has_the_fix():
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_v611 as v611, sf_mynnsfc_v610 as v610, tempo_v325 as t325
    assert mynn6._MY1[1].replace("0.9f", "p9") in v611.MODULE_CUDA and mynn6._MY2[1] in v611.MODULE_CUDA
    assert mynn6._MY1[0] not in v611.MODULE_CUDA and mynn6._MY2[0] not in v611.MODULE_CUDA
    assert v611.MODULE_CUDA.count(mynn6._MY3[1]) == 1 and mynn6._MY3[0] not in v611.MODULE_CUDA
    for old, new in mynn6._MY4:
        assert v611.MODULE_CUDA.count(new.replace("1.f", "one")) == 1
        assert old.replace("1.f", "one") not in v611.MODULE_CUDA
    assert v611.MODULE_CUDA.count(mynn6._MY5[0][1]) == 1 and mynn6._MY5[0][0] not in v611.MODULE_CUDA
    assert v611.MODULE_CUDA.count(mynn6._MY5[1][1].split("\n")[-1]) == 1
    assert "(!(restart||loc_cycle))" in v610.SOURCE and sfc._SF1[0] not in v610.SOURCE
    assert "if (fabs(ssatw[k]) < PS(eps)) continue;" in t325._COLUMN
    assert "T.pbg_rci[k] = PS(meters3_to_liters) * T.prg_rci[k]" in t325._COLUMN
    assert "first && c==0" not in t325._DRIVER and "first && sub==0 && c==0" not in t325._WARP
    for finding, row in fc.FORK_CORRECTIONS.items():
        assert set(row["carried_by"]) <= {"rrfs.bl_mynnedmf_v611", "rrfs.sf_mynnsfc_v610", "rrfs.tempo_v325"}
    assert fc.FORK_CORRECTIONS["TP4"]["carried_by"] == () == fc.FORK_CORRECTIONS["TP5"]["carried_by"]


def test_tp4_on_tempo_v325_changes_only_the_limiter_volume_lines():
    from woof.hex.mpasphys.rrfs import tempo_v325 as t325, tempo_phase_v325 as p325
    # Breakage prevented: a correction leaking into the pinned objects every caller without an id gets.
    assert t325.corrected_column(()) is t325._COLUMN and t325.corrected_column(("TP1", "MY1")) is t325._COLUMN
    assert t325._column_source("strict") is t325._COLUMN
    assert "T.pbg_scw[k] = T.pbg_scw[k] * ratio;" not in t325._COLUMN
    fixed = t325.corrected_column(ALL_TEMPO)
    added = [line for line in fixed.splitlines() if line not in set(t325._COLUMN.splitlines())]
    assert added == ["        // Correct mode TP4 (TEMPO 71588d1): the volume of the riming terms scales with their mass.",
                     "        T.pbg_scw[k] = T.pbg_scw[k] * ratio;", "        T.pbg_gcw[k] = T.pbg_gcw[k] * ratio;"]
    assert len(fixed.splitlines()) == len(t325._COLUMN.splitlines()) + 3
    for corr in ((), ("TP4",), ("TP4", "TP5")):
        for numerics in ("strict", "fast"):
            assert ("pbg_scw[k] * ratio" in t325.cuda_source(DM, max_levels=59, numerics=numerics, corrections=corr)) == bool(corr)
            assert ("pbg_scw[k] * ratio" in t325.warp_source(DM, max_levels=59, numerics=numerics, corrections=corr)) == bool(corr)
        assert p325.phase_source(DM, 59, "", corr) and p325.tail_source(DM, 59, "", corr)
    assert p325.phase_source(DM, 59, "") != p325.phase_source(DM, 59, "", ("TP4",))
    assert p325.tail_source(DM, 59, "") != p325.tail_source(DM, 59, "", ("TP4",))
    assert p325.phase_source(DM, 59, "", ("TP1",)) == p325.phase_source(DM, 59, "")


def _snow_moments(column):
    return column.split("TEMPO_HD void snow_moments(")[1].split("#define TEMPO_SNOW_AB(cc)")[0]


def test_tp5_snow_moments_takes_the_mass_moment_at_bm_s_2_in_both_tempo_generations():
    # Upstream NCAR/TEMPO ace0eae moves `smo2_ = smob` into the bm_s = 2 branch and the fit into the else branch.
    from woof.hex.mpasphys.rrfs import tempo_v325 as t325, tempo_phase_v325 as p325
    for name, exact, fixed in (("v3.0.3", tempo._COLUMN, tempo.corrected_sources(("TP5",))[0]),
                               ("v3.2.5", t325._COLUMN, t325.corrected_column(("TP5",)))):
        before, after = _snow_moments(exact), _snow_moments(fixed)
        test = "  if (bm_s > 2.0f - 1.e-3f && bm_s < 2.0f + 1.e-3f) {\n"
        # pinned: the fit runs when bm_s is 2; corrected: the mass moment is taken there and the fit is the else branch
        assert before.split(test)[1].lstrip().startswith("loga_ = SA(1)"), name
        assert before.count("} else {\n    smo2_ = *smob;\n  }") == 1, name
        assert after.split(test)[1].split("} else {")[0].strip().splitlines()[-1].strip() == "smo2_ = *smob;", name
        assert after.split("} else {")[1].lstrip().startswith("loga_ = SA(1)"), name
        assert after.count("smo2_ = *smob;") == 1 == after.count("smo2_ = tempo_pow((*smob / a_), (1.0f / b_));"), name
        # nothing outside snow_moments moves, and the routine gains one line (the comment)
        assert exact.replace(before, "") == fixed.replace(after, ""), name
        assert len(after.splitlines()) == len(before.splitlines()) + 1, name
        assert sorted(l for l in after.splitlines() if "Correct mode" not in l) == sorted(before.splitlines()), name
    # Breakage prevented: the correction leaking into the objects exact mode and every caller without the id get.
    assert tempo.corrected_sources(()) == (tempo._COLUMN, tempo._DRIVER, tempo._WARP)
    assert tempo.corrected_sources(("TP5",))[1:] == (tempo._DRIVER, tempo._WARP)
    assert t325.corrected_column(()) is t325._COLUMN and t325.corrected_column(("TP4",)) != t325.corrected_column(ALL5)
    # The fast transforms and the phase split still find every anchor on the corrected column.
    mark = "smo2_ = *smob;\n  } else {\n    loga_ = SA(1)"
    for numerics in ("strict", "fast"):
        for mod, corr in ((tempo, ALL5), (tempo, ("TP5",)), (t325, ("TP4", "TP5")), (t325, ("TP5",))):
            assert mark in mod.cuda_source(DM, max_levels=59, numerics=numerics, corrections=corr), (numerics, corr)
            assert mark in mod.warp_source(DM, max_levels=59, numerics=numerics, corrections=corr), (numerics, corr)
            assert mark not in mod.cuda_source(DM, max_levels=59, numerics=numerics)
    for ph, corr in ((phase, ALL5), (p325, ("TP4", "TP5"))):
        assert mark in ph.phase_source(DM, 59, "", corr) + ph.tail_source(DM, 59, "", corr)
        assert mark not in ph.phase_source(DM, 59, "") + ph.tail_source(DM, 59, "")


def test_skip_list_names_fork_ids_and_native_cpu_slot_is_a_named_gap():
    skipped = route_namelist(deck() + PINS + "\n&nhyd_model config_physics_correct_skip='TP2,MY1' /",
                             physics_mode="correct")
    assert skipped.corrections == frozenset({"MY2", "MY3", "MY4", "MY5", "SF1", "TP1", "TP3", "TP4", "TP5", "TP6"}) and skipped.correct_skip >= {"TP2", "MY1"}
    with pytest.raises(ValueError, match="only with config_physics_mode='correct'"):
        route_namelist(deck(), physics_mode="exact", correct_skip=("TP1",))
    with pytest.raises(ValueError, match="unknown"):
        route_namelist(deck(), physics_mode="correct", correct_skip=("TP9",))
    cpu = route_namelist(deck() + "\n&hex_mpasphys config_microp_backend='native-cpu' /", physics_mode="correct")
    assert cpu.corrections == frozenset({"MY1", "MY2", "MY3", "MY4", "MY5", "SF1"})
    gaps = {g["finding"]: g for g in cpu.receipt()["correct_mode"]["gaps"]}
    assert set(gaps) >= {"TP1", "TP2", "TP3", "TP4"} and gaps["TP2"]["backend"] == "native-cpu"


def test_registry_takes_fork_declarations_and_require_catches_a_missing_one():
    config = route_namelist(deck() + PINS, physics_mode="correct")
    registry = SlotRegistry()
    for slot, scheme in config.selectors.items():
        if config.active(slot) and not config.native_embedded(slot):
            declared = {"microphysics": ALL5 + ("TP6",), "pbl": ("MY1", "MY2", "MY3", "MY4", "MY5"), "surface_layer": ("SF1",)}.get(slot, ())
            registry.register(slot, scheme, config.backends[slot], lambda call: None, corrections=declared)
    registry.require(config)
    bare = SlotRegistry()
    for slot, scheme in config.selectors.items():
        if config.active(slot) and not config.native_embedded(slot):
            bare.register(slot, scheme, config.backends[slot], lambda call: None)
    with pytest.raises(RuntimeError, match="do not declare"):
        bare.require(config)
    with pytest.raises(ValueError, match="not fork corrections"):
        SlotRegistry().register("pbl", "bl_mynnedmf", Backend.HEX_CUDA, lambda call: None, corrections=("RP1",))


def test_table_declarations_match_the_scheme_modules():
    from woof.hex.physics_suites import load_table
    for key, mynn in (("mpasphys_rrfs", mynn9), ("mpasphys_rrfs_44c0849", mynn6)):
        slots = load_table()[key]["slots"]
        assert tuple(slots["microphysics"]["cuda_entries"]["mp_tempo"]["corrections"]) == tempo.CORRECTIONS
        assert tuple(slots["pbl"]["cuda_entries"]["bl_mynnedmf"]["corrections"]) == mynn.CORRECTIONS
        assert tuple(slots["surface_layer"]["cuda_entries"]["sf_mynnsfclay"]["corrections"]) == sfc.CORRECTIONS
    assert set(tempo.CORRECTIONS) | set(mynn6.CORRECTIONS) | set(sfc.CORRECTIONS) == set(fc.FORK_CORRECTIONS)


def test_surface_layer_cold_start_condition_is_upstreams():
    assert sfc.SOURCE.count(sfc._SF1[0]) == 1
    fixed = sfc.SOURCE.replace(*sfc._SF1)
    assert "(!(restart||loc_cycle))" in fixed and sfc._SF1[0] not in fixed


def test_state_reflectivity_takes_the_snow_moments_correction_and_no_other():
    """tempo_refl_v303 (the f000 frame's reflectivity) calls snow_moments: with TP5 it compiles the corrected
    routine, on both TEMPO generations; without TP5, whatever else a run applies, its text is the pinned one."""
    from woof.hex.mpasphys.rrfs import tempo_refl_v303 as refl
    mark = "smo2_ = *smob;\n  } else {\n    loga_ = SA(1)"
    assert refl.REFL_CORRECTIONS == ("TP5",)
    for version in ("v3.0.3", "v3.2.5"):
        plain = refl.module_source(59, version)
        assert mark not in plain, version
        assert refl.module_source(59, version, ()) == plain
        assert refl.module_source(59, version, ("TP1", "TP2", "TP3", "TP4")) == plain
        fixed = refl.module_source(59, version, ("TP4", "TP5"))
        assert mark in fixed and fixed != plain, version
        assert fixed == refl.module_source(59, version, frozenset({"TP5"}))
        assert len(fixed.splitlines()) == len(plain.splitlines()) + 1, version


def test_tp6_activation_takes_the_land_mask_in_both_tempo_generations():
    """TP6: the droplet activation gets the cell's land mask (the water floor lower_lim_nuc_frac 0.15 then runs over
    water and sea ice, as the CCPP Thompson call passes lsml). Breakage prevented: the floor is dead code with land
    absent, and marine stratocumulus drizzles into a false light-echo sheet on coarse cells. Exact mode and every
    run without TP6 keep the pinned text; the land word arrives as pointer 20 of the field table."""
    from woof.hex.mpasphys.rrfs import tempo_v325 as t325, tempo_phase_v325 as phase325
    call = "activate_cloud_number(P, temp[k], w1d[k], nwfa[k], C.xland ? &tp6_land : nullptr)"
    for column in (tempo.corrected_sources(("TP6",))[0], t325.corrected_column(("TP6",))):
        assert column.count(call) == 1 and "activate_cloud_number(P, temp[k], w1d[k], nwfa[k], nullptr)" not in column
        assert column.count("  int xland;  // correct mode TP6") == 1
    for column in (tempo._COLUMN, t325._COLUMN, tempo.corrected_sources(ALL5)[0], t325.corrected_column(("TP4", "TP5"))):
        assert "xland" not in column
    # the warp drivers set the word after their context line, with TP3 or without
    for warp in (tempo.corrected_sources(("TP6",))[2], tempo.corrected_sources(("TP3", "TP6"))[2],
                 t325.corrected_warp(("TP6",))):
        assert warp.count("ctx.xland=f[20]?(int)f[20][c]:0;") == 1
    assert t325.corrected_warp(("TP4", "TP5")) is t325._WARP and "xland" not in tempo.corrected_sources(ALL5)[2]
    # the phase prologue (both generations) and the composed modules
    for ph, corr in ((phase, ALL5 + ("TP6",)), (phase325, ("TP4", "TP5", "TP6"))):
        src = ph.phase_source(DM, 59, "", corr)
        assert src.count("C.xland = f[20] ? (int)f[20][c] : 0;") == 1 and "f[20]" not in ph.phase_source(DM, 59, "", ())
        assert "ctx.xland=f[20]" in ph.tail_source(DM, 59, "", corr) and "f[20]" not in ph.tail_source(DM, 59, "", ())
    for numerics in ("strict", "fast"):
        assert "ctx.xland=f[20]" in t325.warp_source(DM, max_levels=59, numerics=numerics, corrections=("TP6",))
        assert "xland" not in t325.warp_source(DM, max_levels=59, numerics=numerics, corrections=())
    assert fc.FORK_CORRECTIONS["TP6"]["carried_by"] == ()


def test_tp6_land_word_is_required():
    """launch_mpas refuses TP6 without the cells' xland (one line naming it) rather than reading pointer 20 past the
    end of the field table."""
    with pytest.raises(ValueError, match="TP6 needs the cells' xland"):
        tempo.tp6_land({}, 10)
