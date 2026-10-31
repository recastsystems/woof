"""Every CUDA kernel source pinned by SHA-256, ONE TEST NODE PER MODULE.

THE BREAKAGE THIS PREVENTS
--------------------------
The 2026-08-28 fault-injection audit disabled a physics process in a CUDA
kernel -- WSM6 rain accretion of cloud water, ``pracw = 0.0f * fminf(...)`` in
``woof/core/kernels/wsm6.cu`` -- and ran the ENTIRE 172-file stage-1 leg
twice, pristine and injected, one pytest process per file:

    pristine tip : 37.5 min, 12 files rc!=0, 13 failing node ids
    injected     : 36.8 min, 12 files rc!=0, 13 failing node ids

Zero files differed.  Zero test node ids differed.  Time to detect: never.

Two separate causes, and this file addresses both.

1.  ``tests/test_mp8_frozen.py`` holds the digests, and it is NOT on
    ``tools/battery/stage1_files.txt``.  The list's own header (line 1026)
    says why: nine kernel modules had drifted from ``FROZEN_MODULE_DIGESTS``,
    so the file is red, and "a known-red file on a per-cut list buys a red
    battery, not a fixed test".  The same header writes "a frozen digest
    nothing checks is not a freeze."

2.  Even run directly it does not DISCRIMINATE.
    ``test_every_frozen_kernel_module_is_unchanged`` collects every drifted
    module into one dict and asserts once, so under the injected fault it
    fails with the IDENTICAL two node ids as the clean baseline.  Only the
    assertion message grows a key.  No exit code, no node id, no count moves.
    An already-red aggregate gate has stopped being a gate.

This file is the same claim at ONE NODE PER MODULE.  A tenth module going
red is a new node id, visible in the count and in the summary, whether or not
the other nine are red.

Measured on this tree at b47a400a5 (and identically at 659962929, the
published 2.5.8 tip): 56 of the 65 pinned modules still match
their historical freeze; nine have moved, each in a named commit, and are
re-pinned below to the bytes they now carry.  Re-pinning is not loosening --
every one of the 65 is pinned to a specific digest, and the drift is written
down with the commit that caused it, which is the amendment discipline the
repository already uses for a stale gate table.  Nothing here is exempt and
nothing here is skipped.

It hashes 115 CUDA translation units and their declared device headers.
It compiles nothing, imports no kernel and needs no CUDA device.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNELS = REPOSITORY_ROOT / "woof" / "core" / "kernels"


def _extension_pins(path=None):
    """Read new scheme authorities, never derive pins from current source.

    Port declarations record measured source digests once in this manifest.
    Existing freeze authorities remain immutable here. An extension cannot
    override them; changing an established identity still requires its
    original authority and evidence to be reconciled.
    """
    path = path or REPOSITORY_ROOT / "tests/data/kernel_source_pins.json"
    document = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "gpuwm-kernel-source-pins-v1":
        raise ValueError("invalid kernel source pin manifest schema")
    for group in ("modules", "headers", "composed_headers"):
        rows = document[group]
        if not isinstance(rows, dict):
            raise ValueError(f"invalid source pin group {group}")
        pattern = r"[A-Za-z0-9_]+" if group == "modules" else r"[A-Za-z0-9_]+\.cu[h]?"
        for name, sha in rows.items():
            if not re.fullmatch(pattern, name) or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError(f"invalid source pin {group}.{name}")
    return document


def _merge_extension_pins(existing, extensions):
    for name, sha in extensions.items():
        if name in existing and existing[name] != sha:
            raise ValueError(f"extension attempts to replace existing pin {name}")
    existing.update(extensions)


def _frozen_module():
    """``tests/test_mp8_frozen.py`` loaded BY PATH.

    ``from tests import ...`` resolves to an unrelated ``tests`` package in
    site-packages on this box -- the same shadowing ARWEN-ORIENTATION
    section 9 records for ``woof``.  The path is explicit and asserted so a
    tripwire cannot end up reading the wrong tree's pins.
    """

    path = REPOSITORY_ROOT / "tests" / "test_mp8_frozen.py"
    assert path.is_file(), f"{path} is missing; its digests are the source"
    spec = importlib.util.spec_from_file_location("gpuwm_frozen_pins", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpuwm_frozen_pins"] = module
    spec.loader.exec_module(module)
    assert pathlib.Path(module.__file__).resolve() == path
    return module


#: ``{module: (file_sha256_now, the commit that moved it)}`` for any module
#: that no longer matches ``FROZEN_MODULE_DIGESTS``.  This table makes a
#: drift explicit and CHECKED -- a further edit to a listed module fails
#: its own node exactly as an edit to a frozen one does -- and the gate
#: beneath makes it TEMPORARY: once a drift is ratified into the freeze,
#: its row here must go.
RE_PINNED_DRIFT: dict[str, tuple[str, str]] = {
    # The three bandwidth modules are ratified into the frozen table
    # by recorded native-word reproduction on their original cards.
}

#: The kernel translation units that ``FROZEN_MODULE_DIGESTS`` never pinned.
#: The freeze deliberately ignores modules added after it was taken, which is
#: right for a FREEZE and wrong for a guard: measured on this tree at
#: b47a400a5 there were twenty-six, and they are not a tidy set of new mp=28
#: additions.  They include ``gf`` (Grell-Freitas cumulus), ``wdm6``,
#: ``milbrandt2``, ``myjpbl``, ``myjsfc``, ``mynn_dmp_sibling``,
#: ``mynn_scalar_mix``, ``noahmp_glacier``, ``shinhong``, ``sase`` and
#: ``tke_budget`` -- shipped physics whose CUDA source no digest anywhere in
#: the repository checked.  These are their bytes at b47a400a5, which is 2.5.8
#: plus the P3 CUDA port.
#:
#: This is a BASELINE pin, and it says so: it does not claim these bytes are
#: right, it claims they are the bytes 2.5.8 shipped, so the next change to
#: any of them is a decision somebody makes on purpose instead of an edit
#: nothing observes.  The comment above each entry is the commit that last
#: moved that file.
#: The shared reason for the 2026-10-05 re-pins of the five thompson_aerosol
#: units and their common header.  WRF v4.6.1's mp_thompson accumulates the
#: cloud, rain and ice tendencies (qcten, qrten, nrten, qiten, niten) across
#: the whole call and applies them once (module_mp_thompson.F :1670 zero,
#: :3975 cloud, :4023-4053 rain and ice, bounded after the freeze); the port
#: applied each stage to the state in place, one extra float32 rounding per
#: stage, which left the g3 gate red on 7 cells since 2.8.5.  The rework
#: (lane/cut286-mp28-g3, 87762b73a + 085c4db97, 2.8.6) carries WRF's
#: accumulators in the v4.6.1 generation; the WRF 3.9 fork takes the cloud
#: accumulator (its own Fortran applies qcten once too, HRRR v4.1.21
#: module_mp_thompson.F:3691) and keeps its in-place rain and ice in its
#: THOMPSON_AA_WRF39 arms; thompson.cu (mp=8) is untouched.
#: Measured 2026-10-05: the g3 gate clears all 22 oracle fixtures inside the
#: flat 2e-6 bound with no allowance on an RTX 5090 and an RTX 4090; a 1 h
#: mp=8 forecast of a 151 x 151 x 50 real case writes byte-identical wrfout
#: files before and after on both cards; the fork generation moves only in
#: its cloud, toward its own Fortran (tools/thompson_fork_oracle/
#: fork_column_parity.py: qc 2.9e-7 -> 1.2e-7, nc 3.4e-7 -> 2.3e-7 relative).
MP28_ACCUMULATOR_REWORK = "lane/cut286-mp28-g3 (2.8.6)"

BASELINE_PINNED: dict[str, str] = {
    # HRRR initial helper: all 3456 native output words, both selectors.
    "gsd_initial_reflectivity": "de41b659db22ffa2262fe52308d402a5de940a3d3c0c326d35c366e07709dd1d",
    # All 48 pinned-fork climate outputs, 106 native gfortran calls.
    # 2.9.7 fold: the minute conversion is __fdiv_rn(dt, 60.f), the A146
    # literal-division rule; an IEEE division, as the gfortran reference divides.
    # Previously 7686eceb.
    "climate_diagnostics": "da46de7a67990141cb1d4a64f2e97672adbce7a01fe911d258ce02312736158d",
    # golden/ports-rad: four translation units that joined with their ports,
    # each graded word for word against WRF v4.6.1's own routines on its lane
    # (the 2026-10-08 port receipts) and re-run at this tip's GPU legs.
    # New Goddard radiation (ra_lw/ra_sw_physics = 5), lane ra5-goddard.
    # Re-pinned in the 2.9.0 round-2 merge: five constant divisions spelled
    # __fdiv_rn (A146; the unit compiles on the -ftz=true route, where NVVM
    # turns x / C into x * RN(1/C) on Blackwell).  Strict arithmetic, where
    # x / C was already IEEE, is unchanged; the device oracle reruns on the
    # GPU shard.
    "goddard": "05d66de119882ea269e05f27469e0ab922fa70dd362d6ab40595870ce6a45776",
    # aer_opt = 2 user AOD/Angstrom optics, lane aer-opt2-aod.
    "aer_opt2": "8f35a7663dd28386c27dbe7c36e614c5da687e58b23e78d20d1fdcb7067e3475",
    # aer_opt = 1 six-type climatology optics, lane aer-opt1-climatology.
    "rrtmg_aer1": "a1b42d2058f6a39f8bc90097f6cda0350d0a82a95dfcb87c4014c3267e0d2d0c",
    # lightning_option 1/2/3/11 diagnostics, lane lightning-param.
    "lightning": "124aea611e07a8e7a6f6c375a2346096552ece2e80cd0d625e4f84dc41d47252",
    # WRF FAST_SBM (mp_physics=30), joined with the port merge for 2.8.9:
    # bit for bit with WRF on every recorded column call (80 of 80 Fortran
    # streams).  These bytes carry the one-column-per-thread launch (64 per
    # block); the public 600 s case and a nested HRRR winter tree write
    # byte-identical history at every block width measured.
    "fast_sbm": "cd0ca83c22d3252bcd669ca16f4b7181c8e151a93a9ec0947777c617c7980881",
    # Generated classic generation; full output-word oracles cover its numerics.
    "thompson_column": "340467a1fb2e50b9b6b8341aeb5aac98a19d0a1abf3afb43d16f4746fcd37be8",
    # Full WRF 4.8 canopy entry, with independent corrected-water selectors.
    "urban_ucm480": "20548fb8212e4bd7bde9c06fa99ebde6da2be28390d7260841a7bb55daa5d701",
    # RE-PINNED 2026-10-15 (lane/291-fix-correct-mode-5, correct mode WB5
    # in WSM7): the rain-out integral gains a WSM7_PLM_SURFACE arm that books
    # the straddling cell's below-ground part with the remap's own PLM
    # profile (RunConfig.wsm7_conservative_surface).  With the macro
    # undefined the preprocessed unit is the 2.9.0 bytes (2c7e45b1,
    # tests/test_correct_mode_water_budget.py strips the arm and compares
    # the sha and the compiled image).  wsm3 and wsm5 moved the same way in
    # tests/data/kernel_source_pins.json (2.9.0: 0daed0b8, 8fbdf5f5).
    # fer_hires moved in that manifest at 18dbe1d5449 (analysis-frame radar
    # reflectivity): an analysis_only launch argument adds a no-source,
    # no-fallout arm that writes REFL from the initial hydrometeors, and
    # the forecast arm's Zsnow/Zrain expressions moved verbatim into the
    # shared fh_snow_z/fh_rain_z inline helpers (same operands, same order);
    # with analysis_only = 0 every forecast branch is the old source.
    "wsm7": "cfa0a7736efc1d755d185bebe1f8f351c469466e844139bc73d7144a44c2d344",
    # Already-staged vertical order kernel (f82847049); lane/ec-pd-upwind-286
    # confined its downstream cell at Courant <= 1 to the strict WRF
    # verification build (it manufactured scalar mass).
    "pd_vertical_sl": "c2b845e06f4aa693f021aadc19d8a32fe001da645f4c48b4684d7c99a09e231d",
    # Already-staged upper-wind source, unchanged by this RUC merge (its
    # baseline bytes came with 4b2bd1665, the fork saved-wind limiter; the
    # AQ line pinned the same digest a second time, merged into this row).
    # Its fork device oracle and generic OFF controls are retained in
    # test_upper_wind_limiter.py; freeze its admitted raw source here.
    "upper_wind_limiter": "7c097a45c226cceda7ecae2424ce8a0fa3cd98a63648a12395be9f29d00ed724",
    # CCPP sa-TKE-EDMF PBL (bl_pbl_physics = 901), joined with its port merge
    # for 2.9.1 (b90138f9d): 64 columns x 48 levels x 3 steps match the pinned
    # upstream CCPP GFSv17.HR4 Fortran (tools/satmedmf_ccpp_oracle) at 0 ULP in
    # both arithmetic modes, re-run on an RTX 5090 at this tip.
    "satmedmf": "9d43f30b081ef35dde5b7992f979c09c5db71d8aa15e5c475ed098e3ed4c5f82",
    # WRF's one-dimensional ocean mixed layer (sf_ocean_physics = 1), joined
    # for 2.9.1: graded against WRF's own module_sf_oml column (112 columns,
    # 16 steps) at 0 ULP in both arithmetic modes (tests/test_oml_gpu.py).
    # RE-PINNED by lane/cap-seaice (7047f0c9bd7, 172be416a1e): oml_step takes
    # the prognostic sea-ice heat and mask and a coupled_ice switch; with
    # coupled_ice = 0 every statement of the open-water column is the 2.9.1
    # one.  Under ice it carries the ice owner's heat and keeps entrained
    # polar water at or above freezing.  Previously 969bc568.
    "oml": "ce97181b17e72439ceb7cab26a9e088013669bb94e42b786b1b14293e4241728",
    # e677dccb6: active table edits multiply binary64 values on the GPU,
    # round once to REAL, and reject non-positive crop roughness.
    "physics_params":
        "7054ef168fe48437d0eeec18f27055959888bd7594af9f38c4b75720c8083e69",
    # Sun-angle albedo: normalization, MODIS classes, first-call copies,
    # retained masked values and the ALBSOL-only cap are compared with
    # extracted fork Fortran at supplied COSZEN. This pin does not qualify
    # the upstream solar-geometry libm seam or forecast skill.
    "solar_albedo":
        "056fb00313f426f294e6da9577e0ed1d05be071f150ec6d1c402b497fbb87d24",
    # The 2.8.5 ensemble line's three translation units, pinned at the bytes
    # that line merged (they joined the product with no digest at all).
    # 234f5ee94: member word-table copies for the resident ensemble batch;
    # tests/test_ensemble_batch_bookkeeping_gpu.py grades it on a card.
    "ensemble_bookkeeping":
        "e323f39729e2248a133f362a5cb5ad0a0c7339c90b3065705919b009602f2d1a",
    # 86dfe06f0: WRF v4.6.1 module_stoch spectral AR(1) patterns over
    # Philox4x32-10; tests/test_ensemble_stochastic_streaming_gpu.py and
    # tests/test_spp_consumers_gpu.py run it on a card.
    "ensemble_stochastic":
        "fed01693da7edf891b8596024cf0244fb560547d292cfa156a1f091c444f35b0",
    # 1ef07b807: WRF V3.9.1 RUC hydraulic-conductivity SPP operator, kept out
    # of ruc.cu so the disabled path keeps its translation unit;
    # tests/test_ruc_spp_gpu.py compares it with the native overlay oracle.
    "ruc_spp":
        "c116898c0bcb37e4b8bf371ca495b445e816d4a1b441fc24c5ba4bb6269fcbb3",
    # Complete WRF v4.6.1 lake driver. The source-pinned native oracle and
    # test_lake_gpu compare all persistent and output words over 300 steps.
    # RE-PINNED at 2.8.5 from 120fa6f73e711147 by the RUC vegetation, albedo
    # and sea-ice lane: lake_init_columns and lake_step_columns take the
    # run's xice_threshold as an argument (module_surface_driver.F
    # :1365-1368 hands lakeini and lake the same 0.5 or 0.02 the land
    # surface runs) in place of the two local ``xice_threshold=0.5``
    # assignments.  At 0.5 every word is the previous build's.
    # RE-PINNED at 2.8.6 from e47718427bb04a9b by the lake arena (A3/A13):
    # both entry points take a launch window (col0, count) and the device
    # arena the column arrays now live in, and set up that thread's arena
    # before the column is built.  No statement of the column changed;
    # test_lake_gpu compares every word against the native oracle, in one
    # window and in three.
    # RE-PINNED by lane/cap-lake (f9eb12244a9): one new entry point,
    # lake_bind_source_columns, binds table-driven lake analyses to the CLM
    # restart layers.  Appended after the existing units; no statement of
    # lakeini or the lake column changed.  Previously 0624bfb3.
    # RE-PINNED by fix/conus5km-32gb (e8b9269ce2, X12): a column the lake
    # source did not cover keeps lakeini's state in lake_bind_source_columns
    # (it took the top layer as T_GRND whenever another lake in the same call
    # was observed); a streamed 5 km CONUS hour then equals the resident hour
    # bit for bit (241/241 variables).  Previously b36729b9.
    "lake": "b57ad46930d5db97b68b37c7e9db81bab469da481f7342ec0141994f693241a8",
    # Native bulk source/units bridge, both GPU receipts; FTZ disabled.
    "chem_sfire": "01a0c2dd43046d33254e4365d4cd87ee472b468d65077dfc06716b9bfe6e6596",
    # lane/283-sfire: compiled WRF v4.7.1 routine controls at 0 ULP
    # on both GPUs; wind/moisture driver and six-composition census
    # receipts in tools/sfire_wrf471_oracle and sfire_coupled_ideal.
    "sfire_atm": "b557a14a9c24d343077187001841e5a7fb5909f93aca7590aba4424e94d6ba80",
    # A captured native quadrature raises one remaining-fuel word by one
    # ULP. The driver preserves consumed fuel before heat and smoke;
    # original and corrected compiled controls are retained separately.
    "sfire_core": "51c0f7a7382a8abdaec33e8154342f3c330b80875150f2ee9367898a2cff94a1",
    "sfire_phys": "e4e217cf69556cb2349ed01e2032be01edba51b94562f0a63525a6eb5c79dcb8",
    "sfire_moisture": "4780e2e58c421c68d7cdfc6f868e25fea4fb2909286e826095fa321edc404ea1",
    "sfire_wind": "81923bd9bf4c9ab11ec240cef898781c1e646520ab2047dc2493c358466435e6",
    "sfire_coupling": '0ce0ae82be28c7a049cd00af4cc540f1eb4ed9594a4972ca10cd4de1cb829754',
    # 47 original/corrected native controls, 96 complete frames, and
    # mapped-host/streamed/disk twins on both GPUs. Eight-composition
    # census and exact receipts accompany the source in the SFIRE tools.
    "sfire_spotting": "f5f6702af0dd58f433ae01502a31aa7c09d15a7bcb5da64c803ca04b570c7a97",
    # Compiled original ideal initialization controls on both GPUs, with
    # separately retained native geometry and table-boundary defects.
    "sfire_ideal": "fb453b222c1915be8bce43f491a5ab634c11ccad6e688644bce27f8711251adc",
    "sfire_ideal_atmos": "4a3676370effcee9c4a4413b1d649b19f25335b798080c770060e27dfa28751a",

    # gp-libm64: new test-only host-library bit grading entry points.
    "portable_libm64_grade":
        "080beeaea2eb617ae3e53beb529f8f99e793eea48c572710d0939c9ec19bb3a9",
    # The card route of initialize_real: portable thermodynamics and the
    # FP32 geopotential split (real_init_math), the arithmetic REAL twins
    # (real_init); bitwise gates on sm_89 and sm_120.
    "real_init_math": "54ab928d5769cde62f8d7bbd8ac56cef36c9cfce719ab45f6b3652638b9af0d0",
    "real_init": "d730af022f5411b04c81ea52ec916254747bafe8677af6393321eb9654aa87e7",
    # Fused horizontal arithmetic and portable binary64 RH conversion.
    "horizontal": "dccca1222ba3e4d7be4157761deecceb00beed9db2c3318bdb5ae683960ed22c",
    # gp-closure: gathered seeding with explicit subnormal rounding.
    "thompson_cold_start": "900fa9a0c368943ade7edcb7ed0ffebc80823ed9136824c235365dd161731548",
    # 679a5f5fe: WRF v4.7.1 coordinate-surface horizontal diffusion.
    # Re-measured after the merged diffusion oracle's boundary-donor fix:
    # the coordinate translation unit is unchanged. Both exports' driver
    # attributes are recorded through the production loader on sm_120.
    "diff_opt1":
        "bbe1360f8ea5249270a89a11f5f2cd77f3fdc573d66d57755a37852ddacaa161",
    # 6c974973d: native cold-start soil liquid water reuses frozen Noah FRH2O.
    # test_wrfinput_cold_start.py drives real kernels against WRF-order
    # startup mirrors; test_kernel_loader_inert.py preserves forecast Noah.
    "noah_init":
        "32ecd84ca52a513304d793664f5c88df9a7904bd50ffa2504a6339c90cdd3897",
    # 6c974973d: terrain-following cold-start W, rounded like set_w_surface.
    # test_wrfinput_cold_start.py retains WRF fill-mode and flat controls.
    "wrf_cold_start_w":
        "c33a86a5f2ff3884ab55265499ef2a93bf59e637461b1a764dbb948720a54303",
    # 3eb49b2fd: evaluation-time moist boundary conversion; measured
    # 73fc2406f evaluator/shared-dycore/public 1800-second forecast.
    # Raw + assembled compiler identities and resources are retained in
    # docs/measurements/lbc-time-2026-09-05/{windows,linux}.json.
    "lbc_time":
        "e98471547d79502806324b840eae701af51d3e6f7e21682d0d7602049513a8c2",
    # 6106e4e31 feat(ftz): measure FP32 subnormal handling on all five compile routes
    "ftz_probe":
        "a8c76a2f19dce3eb9ce545f2caacd1891ac35c3de56c7d57e39e787f37e1717f",
    # d0c23dad0 feat(cumulus): New Tiedtke joins as cu_physics = 16 -- the glibc
    # float32 transcendentals leave gf.cu for the shared glibc_flt32.cuh the
    # loader prepends; a move, not an edit (517 non-trivial lines out, 517 in)
    # RE-PINNED at 2.7.0 by the gamma replacement.  What moved, and the
    # measurement that says the new behaviour is right:
    #   * ArWen's earlier gamma is gone from the header this unit
    #     prepends, replaced by ArWen's own correctly rounded gamma, which is
    #     CORRECTLY ROUNDED on all 59,768,833 float32 of [0.25, 36] where
    #     glibc 2.39 is not on 23,575,230 of them (39.4440 per cent, worst 6
    #     ULP).  Measured against a 113-bit tgammaq oracle;
    #     tests/test_gf_gamma_correctly_rounded.py is the gate.
    #   * gf_libm_unary_probe drops from 7 slots to 4: gfk_lgamma_pos,
    #     gfk_expm1 and gfk_exp2 were reached only by that gamma block and
    #     are deleted with it.
    #   * gf_gfdrv_stage gains three scin slots (DINS_fzu_up/dn/sh), the
    #     same fzu_override gf_deep_stage and gf_shallow_stage already
    #     expose.  The shipped forecast passes 0 in all three; the parity
    #     suites pin them from the WRF capture, which is what keeps the
    #     216-column GFDRV boundary bitwise (host crosscheck: 83/83 level
    #     fields, 69/69 scalars, 39/39 integer fields, 126/126 shallow,
    #     0 driver gate failures).
    #   * the physics change is a DELIBERATE DIVERGENCE and is written up in
    #     docs/gf_gamma_known_delta.md: RAINCV/PRATEC move by at most 7.273
    #     per cent, median 1.613, on the committed capture, and no integer
    #     index field moves on any of the 216 columns.
    # RE-PINNED by the WRF-parity cup_up_aa0 repair (PAR-CU-GF-02).
    # module_cu_gf_deep.F:3024 is `IF(K.LT.KBCON(I))GO TO 100`, so the level
    # k == KBCON contributes to the undilute CAPE; the kernel wrote
    # `k <= kbcon` and dropped it.  The index convention is settled by the
    # sibling cup_up_aa1bl (:4048 `IF(k.gt.KBCON(i))`, gf.cu:1168
    # `if (k > kbcon) continue;`), and woof.verify.gf_deep_ref:971 already
    # read `k < kbcon`.  Answers move only where the dropped level is the
    # whole of aa0: the committed 216-column oracle
    # (tests/test_gf_wrf461_parity.py) is byte-identical either way and
    # stays 2/2 green.  Measured by tests/test_gf_workspace.py::
    # test_cup_up_aa0_keeps_the_k_equals_kbcon_layer, which compiles gf.cu
    # and drives the device function through ctypes: 0.0 before, 0.0348961316
    # after, identical to the Python reference.  gf is not an mp=8
    # translation unit and thompson.cu is byte-unchanged.
    # Composed licence + parity source independently checked before this pin:
    # gamma CPU/device gates and the kbcon-layer probe: 17 passed
    # on 2026-09-04. Neither standalone branch digest names these bytes.
    # RE-PINNED at 2.7.6 from 334c55ab764732be by the notice correction: the
    # comments at lines 62-65 and 2820 stop describing the earlier gamma as
    # derived from glibc.  Comments only, measured: with // and /* */
    # comments removed the file is byte-identical to the 334c55ab bytes, and
    # it keeps its 4,061 lines, so every line number cited above still holds.
    # RE-PINNED 2026-09-30 by the default-pieces speed lane (b11fc66ab): a
    # production column the first trigger rejects writes its zero outputs
    # at once, the environment above the trigger's reach is built only for
    # columns that pass it, and the driver launch is bounded to eight
    # resident blocks per SM.  No answer moved: the lane's A/B dumps after
    # 23 steps (sm_120) and a 1 h real HRRR GF forecast (sm_89) are
    # byte-identical to d6929cb8d.  Previously 2ca7ac7bbeb01627.
    # RE-PINNED by lane/290-A5-fixes (exact chain 7cdf42951): CUP_gf keeps
    # descending after a valid downdraft-origin adjustment instead of
    # breaking out of the pass, as WRF does; it moved the strict A071 hour's
    # first wind difference from step 87 to step 169 against stock WRF
    # 4.6.1 (SOL-POOL-2026-10-09 exact-chain-resume).  Previously ad4148a8.
    "gf":
        "120dc7e606dae18cece64e1d1af899ec8930de339e82943dcab3457c9115532d",
    # 1ee7f0be0 tiles: name the streamed-run config table, and part it from cycle streaming
    "health_tile":
        "2943d5e226a61487aefbe7f191dc120420a4cfe3f96deef19c90c2bb8c15bead",
    # d76e25a82 feat(da): the LETKF factors its own matrices; cuSOLVER becomes optional.
    # Re-pinned by lane/ec-da-science-eigh: the JACOBI_GLOBAL_WORK tier for
    # k above 64 (ensembles of 65 to 256 members stay on this kernel instead
    # of cuSOLVER); without the define the preprocessed shared tiers are the
    # same lines, and tests/test_jacobi_eigh_gpu.py proves the forced global
    # tier reproduces them bit for bit.  Previously 7e24eff5cf84ff68.
    "jacobi_eigh":
        "d2b39cfd26606a85b5f810bfca9fc576d39f8b33a9f256531b4a64f76978bcbd",
    # 0c8f2305d Batch native KF output validation
    "kf_validation":
        "697a1cab3ab07d2e1464c03cad72c08bda809d461a33b5d67273e31ca2a71f56",
    # 5b912c2b9 Batch canonical microphysics validation; re-pinned by the
    # speed lane's one-launch ring guard (mp_ring_copy joins the unit: word
    # copies only, frame 0 B measured on sm_120 at NVRTC 13.4.92; every
    # two-moment and Thompson forecast byte-identical to d6929cb8d)
    "microphysics_validation":
        "9a40f5bda9065d98d7145dbda3473d22ad00a890aad223e5f2e913b16b547b28",
    # d0c23dad0 feat(cumulus): New Tiedtke joins as cu_physics = 16 -- the scheme's
    # translation unit, bitwise against WRF v4.6.1 at every stage (tests/test_ntiedtke_*)
    # RE-PINNED 2026-09-30 by the default-pieces speed lane (5310e64b2):
    # cutypen clears parcel scratch only for trials that can run and the
    # hoist checks are read once per call from integer device masks.  No
    # answer moved: the lane's A/B dumps (sm_120, the fused 350 x 200 path
    # included) and a 1 h real HRRR New Tiedtke forecast (sm_89) are
    # byte-identical to d6929cb8d.  Previously 06daa934a71a0279.
    # RE-PINNED 2026-10-07 by the New Tiedtke column-oracle lane
    # (tools/ntiedtke_column_oracle): answers MOVE, toward WRF.  cuflxn's
    # sub-cloud evaporation and the deep closure's CAPE sum now associate
    # left to right as the reference does, and mfub/closure read ztmst
    # instead of the model dt.  Bitwise against WRF v4.6.1 (zcbase declared
    # real) on 1.1 M real columns; pinned by tests/test_ntiedtke_column_oracle.py.
    # Previously 17fe4aac03ed6b51.
    "ntiedtke":
        "5cc5944d841ec39a0869bbe340488fe0d2995e86e56864ba6437d0f0d1b6f20b",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED by the speed lane's category-parallel sedimentation (no
    # reading moves): five thread rows per column block run rain, ice, snow,
    # graupel and hail at once (disjoint fields), then one thread per column
    # takes the snow constraints and precipitation totals in the original
    # order after a block barrier.  The mp9 suite's 1 h forecasts of two
    # convective cases (220 x 176 x 49) are byte-identical to d6929cb8d on
    # an RTX 5090.
    # Re-pinned for A146 on top of that: constant-divisor float divisions
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal on Blackwell targets. Previously 8b4219be.
    # RE-PINNED by lane/verify-milbrandt2mom-mp9: WOOF's own gfk_exp/gfk_log/
    # gfk_log10/gfk_pow in place of CUDA's libm, gfortran's evaluation of
    # powers (REAL**2.0 a multiply, integer powers by the power tree, powers
    # before products), WRF's *idt where the port divided by dt, NINT as
    # round-half-away.  Measured: every output word of WRF v4.6.1's own
    # mp_milbrandt2mom_driver on 603 oracle columns, default and strict
    # builds (tests/test_milbrandt2_wrf461_exact_gpu.py).  Previously
    # 654a4fd0.
    "milbrandt2":
        "bcc94779ca1e370984e653aa1d89f073d14164471dae4adea1cba12394156cf1",
    # 6933d5762 fix(physics): Milbrandt-Yau joins the radar operator -- the pure
    # Z block of mp_milbrandt2mom_main's final diagnostics
    # (module_mp_milbrandt2mom.F:3400-3466), lifted out of the byte-frozen
    # milbrandt2.cu so woof.da.obsop can launch H_Z(x) without the scheme's
    # state update.  It shipped with nothing checking its bytes at all, which
    # this pin closes; the duplication is held equal to the original by
    # tests/test_da_obsop_milbrandt_gpu.py::
    # test_the_operator_is_the_schemes_own_z_block_bitwise.
    # RE-PINNED by lane/verify-milbrandt2mom-mp9: LOG10 is gfk_log10, the
    # scheme's own, so the operator stays bitwise with milbrandt2.cu's Z
    # block.  Previously dfab83f0.
    "milbrandt2_zet":
        "908c8ab88bc3941ef601154d98b926f11bdc28ee4d9e9a6f780e8d670c5200ba",
    # f1e9adbf1 fix(myj): seed TKE_MYJ at WRF's EPSQ2, make the mutation controls real, decl
    # RE-PINNED by lane/verify-myj (WRF v4.6.1 MYJPBL column oracle,
    # tests/test_myjpbl_wrf461_parity.py, bit-identical under the strict
    # build): ZINT(KTE+1)=HT seed (ht_a argument), gfk_exp_fma/gfk_pow in
    # place of CUDA's expf/powf, B1**(2./3.) and every constant as the word
    # gfortran folded.  Previously d0e0b3dd.
    "myjpbl":
        "267d8d475743fb454ab4715d88bf9007ec3e41501e7e40c765ea4beb9ddb9100",
    # f1e9adbf1 fix(myj): seed TKE_MYJ at WRF's EPSQ2, make the mutation controls real, decl
    # RE-PINNED by the WRF-parity PBLH seed repair (MYJ-01): the height
    # accumulator seeded from ``dz_a[0]`` -- layer 0 of COLUMN 0 -- for every
    # thread, where module_sf_myjsfc.F:177-184 accumulates ZINT strictly
    # inside column I.  PBLH was wrong by ``dz[0][0] - dz[0][col]`` on every
    # column but the first, and it re-enters SFCDIF as BTGH (:431-435).
    # This MOVES ANSWERS on any domain with terrain.  Measured by
    # tests/test_myj_port.py::
    # test_the_surface_kernel_pblh_uses_each_columns_own_dz (two columns,
    # different dz[0] -- the old gate ran at shape (1,1), where col is
    # always 0 and the two indices coincide) and, device-free, by ::
    # test_the_surface_kernels_pblh_accumulator_is_column_local.
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 334ae702.
    # RE-PINNED by lane/verify-eta-similarity-sfclay, MOVES ANSWERS for
    # sf_sfclay_physics = 2: ZINT(KTE+1)=HT seeds the interface heights as
    # module_sf_myjsfc.F:165 does (PBLH and ZSL are differences of them),
    # LOG/EXP/** are WOOF's gfk_log/gfk_exp_fma/gfk_pow, and the unit
    # compiles without FMA.  Every INOUT and output word of 640 column-calls
    # is bit-identical to WRF v4.6.1's MYJSFC under the strict build and
    # default arithmetic (tests/test_myjsfc_wrf461_parity.py).  Previously
    # 7ce67cdd; then d6a84892 while REAL**REAL was a contracted pow, now
    # gfk_pow (same oracle result).
    # RE-PINNED by lane/cap-waves-couple (b8b94dcf810): the sea-point flux
    # iteration consumes the wave model's roughness, compiled in only under
    # OCEAN_WAVE_ROUGHNESS (woof/core/myjsfc.py defines it only when an
    # ocean_z0 field is passed); without it the preprocessed unit is the
    # previous one.  Previously 840d7099.
    "myjsfc":
        '7cb49b17e31cdf4890adece1a5e4b95facaccecb67fe39ac7221174122e61005',
    # Wave3: mynn_dmp_sibling is a logical export-arm alias of mynn_pbl,
    # pinned in test_mp8_frozen.py and test_mynn_dmp_sibling.py.
    # 4a0bb3f69 mynn(mixscalars): MYNN-EDMF mixes the qn family, and the DMP unit exports it
    # RE-PINNED by the 2026-09-30 MYNN speed lane (2f5dd16f9): the flux kernel reads
    # level-major plume and column storage.  Addressing only; the
    # mixscalars replay is bitwise identical.
    # Added WRF diff4d/diff/invert for scalar_pblmix, which uses a separate
    # entry point. test_scalar_pblmix_gpu compares its output words with
    # compiled WRF and exercises consumption by the actual RK3 path.
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    "mynn_scalar_mix":
        "abc61827d645ce7af664a7294ecfd289834de5cc57b485bff20e20d27ed17c1f",
    # 342f8780d feat(glacier): NOAHMP_GLACIER ported, the sea-ice threshold configurable, na
    # RE-PINNED by the WRF 4.7 Noah-MP night snow-aging port: SNOW_AGE_GLACIER
    # runs before the COSZ test under wrf_470 (both generations graded by
    # tests/test_noahmp_snow_age_gpu.py).
    "noahmp_glacier":
        "14c6e57f2a2df7926a05101c89f2a6778b150ce68af58c78996e288bda8cd954",
    # 2258c85e3 feat(p3): mp=50 runs on the card, because a host round trip per
    # step was the scheme.  Added by the P3 CUDA port AFTER 2.5.8 published; it
    # is the module this gate named unprompted the day it was written, which is
    # the case for the reverse test at the bottom of this file.
    # RE-PINNED by the 2.6.1 cold-start qvs-floor fix: the floor lands
    # default-on in all three arms (2026-08-31) -- step-1 sup/supi pin
    # at -1 instead of the stock 0/0 NaN, inert from step 2 on; the
    # measurement is tests/test_p3_port.py::
    # test_the_first_step_qvs_floor_pins_sup_at_minus_one plus the
    # from-step-2 off-path identity.
    # RE-PINNED AGAIN by the immersion-freezing overflow rescue folded off
    # p3/front-door-20260829: the droplet and rain branches gained WRF's own
    # commented-out double excursion, reached ONLY when the single-precision
    # chain overflows to +Inf.  A real 6 h GFS forecast on the shipped
    # p3-mp50 suite went non-finite at step 284 because it does (see the
    # block comment at the branch and woof/core/p3.py
    # _rescue_overflowed_product).  The float path is byte-unchanged, so
    # this is a new branch, not a moved answer.
    # RE-PINNED by the speed lane's level arms (no reading moves): four
    # threads per column share the level-local work of preparation,
    # process rates, sedimentation substeps and final diagnostics, each
    # level's statements unchanged; ``cuda`` now selects ``sedlevels``.
    # 1 h P3 forecasts of two convective cases (220 x 176 x 49) are
    # byte-identical to d6929cb8d on an RTX 5090 and an RTX 4090.
    # RE-PINNED again by the level-group shapes (no reading moves):
    # preparation runs eight groups of 16 columns and finishes both of a
    # level's stages before the next, process rates sixteen groups of 8,
    # sedimentation sixteen groups of 4 (fmaxf Courant maximum, exact).
    # The same two P3 forecasts stay byte-identical on an RTX 5090.
    # Re-pinned for A146 on top of that: constant-divisor float divisions
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal on Blackwell targets. Previously ebbd1f28.
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    "p3":
        "f69de138d6f38e7a4aa12df93e20a9d6bf8e32f9ef8c2c92b283c4292ff80958",
    # Audit S14: P3's radar operator H_Z(x), the Z half of p3_final_level
    # replayed on local copies over p3.cu's own device helpers so it updates
    # nothing; compiled after p3.cu in its own module
    # (woof.core.p3_device.p3_reflectivity_module).  Held to the forecast
    # step's own zdbz by tests/test_p3_reflectivity_gpu.py.
    "p3_zdiag":
        "f22e039d18c19625e1a100b79a6f3ccf0227bad1960215be0d7ecc0cf9e1cbad",
    # 9c57c4ee9 feat(sase): CUDA mirror of the S3-12 additive e^{3/2} dissipation channel, p
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 9c49c1d0.
    "sase":
        "af8c732fbc4945c517a965a7dc4c2e931667400064d01072f27728f4a67d7ae8",
    # a084e0aeb fix(shinhong): the ULP table moved because the kernel compiler did, so it is
    # RE-PINNED 2026-09-30 by the default-pieces speed lane (878435c39):
    # the column arrays move from a 17,160 B per-thread local frame into a
    # global workspace sized to the columns in flight (the YSU route).
    # Placement only; no arithmetic line moved.  The lane's A/B dumps
    # (sm_120) and a 1 h real HRRR Shin-Hong forecast (sm_89) are
    # byte-identical to d6929cb8d, and the ULP table gained its NVRTC
    # 13.4.92 sm_89 and sm_120 rows.  Previously 342e8c86f16262bf.
    # Re-pinned for A146 on top of that: constant-divisor float divisions
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal on Blackwell targets. Previously be8bf6ea.
    # RE-PINNED by lane/parity-pbl-libm (sweep row 20), MOVES ANSWERS for
    # bl_pbl_physics = 11: glibc powf/expf (gfk_pow/gfk_exp), WRF's **h1
    # powf at the former cbrtf site, tanh in double, and --fmad=false.
    # Reading (RTX PRO 6000, NVRTC 13.4.92 and 12.9.86 identical):
    # tests/test_shinhong_wrf461_parity.py goes bitwise on every lane but the
    # 72 flushed subnormal tendencies (du 93207, dv 46603, tke 2022, el 13,
    # exch_h 8, hpbl/wstar/delta 1 -> 0), partition probe 2 -> 0.
    # Previously 6fd615d0.
    # RE-PINNED by lane/verify-shinhong (2026-10-07), MOVES ANSWERS for
    # bl_pbl_physics = 11: thx is formed as WRF forms it, (theta*exner)/exner
    # (module_bl_shinhong.F:559 on phy_prep's t = th*pi), instead of the
    # model's theta read directly.  Reading on the wide column oracle
    # (tests/test_shinhong_column_oracle.py, 903 columns, RTX PRO 6000,
    # NVRTC 12.9.86): strict build from ~1,100 differing lanes per tendency
    # and hpbl in 77 columns to every arithmetic lane bitwise.
    # Previously 26a78c8f.
    # RE-PINNED by lane/verify-topo-wind (2026-10-07): the column body is
    # shinhong_column_body<TOPO>; shinhong_column (arm A) is the same
    # arithmetic, and the new shinhong_column_topo (arm B) is WRF's
    # topo_wind arm (module_bl_shinhong.F:1387, :1472-1473), which this
    # engine refused until then.  Reading: tests/test_shinhong_column_oracle.py
    # and tests/test_shinhong_wrf461_parity.py unchanged (arm A), and
    # tests/test_topo_wind_wrf461_chain_parity.py's Shin-Hong consumer
    # bitwise on every defined lane under the strict build (RTX PRO 6000,
    # NVRTC 12.9.86).  Previously 852f71a6.
    # And csfac's tanh (module_bl_shinhong.F:847) is sh_tanhf, WOOF's own
    # float32 tanhf held to the reference platform's (glibc 2.39), where it
    # was the correctly rounded tanh in double.  MOVES ANSWERS for
    # bl_pbl_physics = 11 on the columns whose csfac argument lands on a
    # differing word: the topo_wind chain sweep's 16 cells.
    "shinhong":
        'cd650676df874db77b0c52303fe6ad3f37a821d3d818d6423b73f3878e46948b',
    # 58dfd599c feat(shinhong): CUDA mirror on the RTX 5090 -- dtheta bitwise, br DAZ counte
    "shinhong_validation":
        "e3714616c403a3f272600499164cf9b0215879d567dadcda31287dd33c600b87",
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: 5 micron crystals for number-less entry ice
    # (:1855-1858), the snow-cloud table's zero 6 micron bin (:4936),
    # the D0i minimum crystal mass (:2649, :2713), the graupel
    # sublimation number gate (:2703), the source-stage rain and
    # graupel balances (:3067-3091, :3118-3160); 217e84e18: the running
    # vapour carried unfloored (:3974); 08f1f9373: the ice mass/number
    # balance above 0 C (:3033-3055).
    # Previously ff9efd45815288a6.
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 42fcf4c2.
    # RE-PINNED 2026-10-03 by the operational WRF 3.9 fork's Thompson
    # (RunConfig.thompson_version = "wrf_39_noaa", lane/286-fork-thompson):
    # every fork statement sits in a THOMPSON_AA_WRF39 arm, compiled only
    # under that integer define; with it absent the v4.6.1 code compiles as
    # before.  Measured: the host parity report against WRF v4.6.1 on the
    # 42 committed real-data columns is byte-identical before and after,
    # tests/test_thompson_wrf39.py strips the fork arms back to the pre-fork
    # source, and the wrf_461 KS/CO cut forecast is compared word for word
    # with the branch base on a card (lane report).
    # Previously d8b8faa7ac626c96.
    # RE-PINNED 2026-10-05 by the mp=28 accumulator rework (lane/cut286-mp28-g3,
    # 87762b73a + 085c4db97, shipped in 2.8.6; MP28_ACCUMULATOR_REWORK below
    # holds the shared reason and measurement).  Here: the cold network
    # writes the accumulators (:2987), its rain, ice and number source
    # balances take WRF's tendency form (:3022-3091), and the rain
    # conservation ratio is REAL as WRF declares it (:1615).
    # Previously 001c8655d12a2f4a.
    # RE-PINNED 2026-10-08 (gfix/288-thompson-classic): the dead slab-five
    # rain-graupel copy after the v4.6.1 dispatch's return (inside if (false)
    # under THOMPSON_AA_WRF39) is removed; production reads aaf_racg_index.
    # Measured on an RTX 4090 (sm_89): the compiled module differs from the
    # previous source's only in .strtab (symbol names carry the source hash),
    # every .text and .nv.info section byte-identical, for both the v4.6.1
    # and the WRF39 generation.  Previously d7ae2da9db4fe7b6.
    "thompson_aerosol_cold":
        'b8f5496b68a29ee0de0a93203a8619175f428040c634bcd099e2e21bab1e49e7',
    # 0ebda6608 snapshot(mp28): the recovered aerosol-aware Thompson port, re-parented to it
    "thompson_aerosol_probe":
        'a4db6f915e5393c2a809f40c0b793ccba5e9bba51a672f7d5ea99a04c1a18274',
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: the adjustment exports L_qc (:3485) and the rain
    # evaporation writes L_qr and the :3568 rewrite into the rain
    # reference density (:3236); 217e84e18: the running vapour carried
    # unfloored (:3974).
    # Previously b54711ac9e07dfe8.
    # RE-PINNED 2026-10-05 by the mp=28 accumulator rework
    # (MP28_ACCUMULATOR_REWORK below).  Here: the condensation (:3480) and
    # the rain evaporation (:3562-3564) write the accumulators and read the
    # working state WRF reads (:3215, :3236-3238).
    # Previously 44d3fb82ecfc49ae.
    "thompson_aerosol_sat":
        '88ee26846dde74b282b286802d0cd838f3332048c74b295cc3723109b2847a30',
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: the terminal ice bound in its per-kilogram form
    # (:4024-4040) and the terminal bounds on the density the rain
    # evaporation left (:3572); c4f3fcc70: cloud at or below 1e-12
    # kg/kg carried to the phase cleanup (:3943-3966).
    # Previously d234db58a7d3cbb1.
    # RE-PINNED 2026-09-30 by the speed change that gives an empty column
    # an exact short path through the aerosol cloud fallout and appends
    # the level-parallel cloud fallout; no answer moved (bit tests and a
    # byte-identical 1 h mp=28 forecast on an RTX 4090).
    # Previously 8380f654e902a4ec.
    # RE-PINNED 2026-10-03 by the operational WRF 3.9 fork's Thompson
    # (RunConfig.thompson_version = "wrf_39_noaa", lane/286-fork-thompson):
    # every fork statement sits in a THOMPSON_AA_WRF39 arm, compiled only
    # under that integer define; with it absent the v4.6.1 code compiles as
    # before.  Measured: the host parity report against WRF v4.6.1 on the
    # 42 committed real-data columns is byte-identical before and after,
    # tests/test_thompson_wrf39.py strips the fork arms back to the pre-fork
    # source, and the wrf_461 KS/CO cut forecast is compared word for word
    # with the branch base on a card (lane report).
    # Previously cff91a9693c01ab2.
    # Comment-only re-pin: the source note omits a personal attribution.
    # Stripping the fork arms still restores the pre-fork default source.
    # RE-PINNED 2026-10-05 by the mp=28 accumulator rework
    # (MP28_ACCUMULATOR_REWORK below).  Here: the cloud fallout (:3832) and
    # the phase cleanup (:3949-3964) add to the accumulators, and mp=28 gets
    # its own tendency-form rain and ice fallout
    # (thompson_aa_{rain,ice}_sediment_accumulate_{64,256}, :3611-3698,
    # :3790-3870), so the classic fallout kernels mp=8 runs stay as they are.
    # Previously 248d83895eacbbe6.
    # RE-PINNED 2026-10-06 by the two reworks together, merged for 2.8.7
    # (lane/da-line-287): the accumulator rework above, plus the fork's rain
    # fallout (audit T21, fork :3556, lane/da-line 1c5693862), a copy of
    # thompson.cu's in-place rain-presence pass appended inside a
    # THOMPSON_AA_WRF39 arm that counts surface rain above R1*10.  The fork
    # keeps its in-place rain under the rework, so the two touch disjoint
    # code: the bytes are the rework's file (52c3df61d73aa53d) with the
    # fork arm appended, and stripping the fork arms still restores the
    # pre-fork default source.  Measured on the merged tree on an RTX 5070
    # Ti: tests/test_thompson_wrf39_rain_fallout_gpu.py (the fork's column
    # state word for word against the frozen pass, its export differing
    # only on trace surface rain) and the mp=28 oracle and sed GPU decks at
    # their existing bounds.  The DA line's own pin was 664ae49ee658def6.
    "thompson_aerosol_sed":
        'bcd106943cec93f8faa8b6281b1bb8d3d9e90888b9b3a4b4e9dc44466bf58019',
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # 217e84e18: WRF's no-microphysics column exit (:1646, :2020), the
    # new thompson_aa_micro_columns flag kernel, and the terminal apply
    # flooring vapour at 1e-10 in every other column (:3974); the final
    # vapour, nwfa and nifa have no unexplained cell on any frame.
    # Previously 856c00e10f3fb4cf.
    # Re-pinned 2026-10-03 for optional stores of unused entry diagnostics.
    # RTX 4090: 77 entry/memory GPU tests passed, including three calls on
    # all 22 fixture states against the full-output path, bit for bit.
    # NVRTC 13.4 compiler census: no constant-divisor or rewrite sites.
    # RE-PINNED 2026-10-03 by the operational WRF 3.9 fork's Thompson
    # (RunConfig.thompson_version = "wrf_39_noaa", lane/286-fork-thompson):
    # every fork statement sits in a THOMPSON_AA_WRF39 arm, compiled only
    # under that integer define; with it absent the v4.6.1 code compiles as
    # before.  Measured: the host parity report against WRF v4.6.1 on the
    # 42 committed real-data columns is byte-identical before and after,
    # tests/test_thompson_wrf39.py strips the fork arms back to the pre-fork
    # source, and the wrf_461 KS/CO cut forecast is compared word for word
    # with the branch base on a card (lane report).
    # Previously 005bdb57bfc83b37.
    # Re-pinned for fork-only start emission, measured against the fork's
    # thompson_init on five cell spacings by test_thompson_wrf39_init_gpu.
    # The default state's 39 GPU gates pass, and the 42-column adapter's
    # 26 outputs remain byte-identical to 7ab2e3dcf. Fork compiler census:
    # zero constant-divisor and rewrite sites in all five translation units.
    # RE-PINNED 2026-10-05 by the mp=28 accumulator rework
    # (MP28_ACCUMULATOR_REWORK below).  Here: the terminal rain and ice
    # apply with WRF's bounds after the freeze
    # (thompson_aa_terminal_rain_ice, :4023-4053).
    # Previously 009f1debca0c3ecb.
    "thompson_aerosol_state":
        'c16b1552e1e0a0980aa83e9a3c197e25b71350521ce71de6badb136dd031e3e9',
    # c1563f187 fix(release-scan): the gate reads by content, and sees an escaped path
    # RE-PINNED 2026-09-24 by the WRF v4.6.1 real-column repairs.  What
    # moved is WRF's own rule in each case, cited to module_mp_thompson.F
    # in the kernel, and the measurement that says the new behaviour is
    # right is tools/thompson_real_column_parity: WRF's Fortran run beside
    # the port's kernels compiled for the host on 137,200 columns of seven
    # saved real-data states, where every mp=28 process rate now agrees
    # to float32 rounding (17 rates had differed beyond 1 percent at up
    # to 2,457 levels a frame) and the echo within 0.024 dB (up to 8.7 dB
    # off in 505 to 939 cells of every forecast frame before); the
    # committed fixture tests/data/thompson_real_columns_wrf461.npz holds
    # it in tests/test_thompson_real_column_host_parity.py.
    # ce303c3e5: the source-stage rain and graupel balances
    # (:3067-3091, :3118-3160); 217e84e18: the running vapour carried
    # unfloored (:3974).
    # Previously 031fa75543cb9408.
    # Re-pinned for A146 (a98f2482e): constant-divisor float divisions
    # spelled __fdiv_rn.  Previously 44621264.  Re-pinned again by the
    # A146 review repair: the two lamc clamps divide by a header constant
    # (THOMPSON_AA_D0C, THOMPSON_AA_D0R * 2.0f), which compute_120 turns
    # into a reciprocal multiply; they now go through thompson_aa_div.
    # RE-PINNED 2026-10-03 by the operational WRF 3.9 fork's Thompson
    # (RunConfig.thompson_version = "wrf_39_noaa", lane/286-fork-thompson):
    # every fork statement sits in a THOMPSON_AA_WRF39 arm, compiled only
    # under that integer define; with it absent the v4.6.1 code compiles as
    # before.  Measured: the host parity report against WRF v4.6.1 on the
    # 42 committed real-data columns is byte-identical before and after,
    # tests/test_thompson_wrf39.py strips the fork arms back to the pre-fork
    # source, and the wrf_461 KS/CO cut forecast is compared word for word
    # with the branch base on a card (lane report).
    # Previously 633791f61719e99f.
    # RE-PINNED 2026-10-05 by the mp=28 accumulator rework
    # (MP28_ACCUMULATOR_REWORK below).  Here: the warm network writes the
    # accumulators (:2987) and its rain balances take WRF's tendency form
    # (:3058-3091).
    # Previously 56abc1c1082ee1f1.
    # RE-PINNED 2026-10-08 (gfix/288-thompson-classic): the dead slab-five
    # rain-graupel copy after the v4.6.1 dispatch's return (inside if (false)
    # under THOMPSON_AA_WRF39) is removed; production reads aaf_racg_index.
    # Measured on an RTX 4090 (sm_89): the compiled module differs from the
    # previous source's only in .strtab (symbol names carry the source hash),
    # every .text and .nv.info section byte-identical, for both the v4.6.1
    # and the WRF39 generation.  Previously 98aa971486192c17.
    "thompson_aerosol_warm":
        '3e6972a7e5c41238c6ba664a8032851c187d007a4bf6366b17020b6dbed9635a',
    # 02cfd5301 feat(les): km_opt=2 restart carrier, lateral-boundary arm, TKE budget
    "tke_budget":
        "c7f6dc37f15b25fccbea50deef0c6d595c08b2ee4762f14eef169b654d54fccb",
    # The UW moist-turbulence PBL (bl_pbl_physics = 9, lane/europe-uw-pbl):
    # every output word of the product launcher equals WRF v4.7.1's on the
    # packaged oracle fixtures (tests/test_uwpbl_launcher_wrf471_parity.py).
    "uwpbl":
        "79ba54c179dbe7d3c7966de5e08d5d223eaf2c169fca02ba864f3baba0ec2423",
    # Dudhia shortwave (ra_sw_physics = 1, lane/verify-dudhia-sw): every
    # RTHRATEN, GSW and SWDOWN word equals WRF v4.6.1's on the packaged
    # column oracle (tests/test_dudhia_sw_wrf461_parity.py), strict and
    # default arithmetic.
    "dudhia_sw":
        "00108722243f47127adbe0e1ee97ecebbf6a15972ff59338e6b52dd08dc2596a",
    # RE-PINNED 2026-09-20 after three commits moved the file and none
    # re-pinned it (red on every box since 2026-09-13; proof/node-reds-276):
    #   * 07d1ef7e3 (2026-09-12) fix(physics): publish current WDM6
    #     checkpoint identity -- the published registry row and the
    #     kernel's identity comment;
    #   * 0a7fc061f (2026-09-12) fix(physics): bound evolving WDM6 rain
    #     transport time -- every receiving layer covered by the existing
    #     fall-speed envelope, conservative interface transfers kept,
    #     temporal refinement verified independently
    #     (tests/test_wdm6_sedimentation.py, docs/wdm6_oracle_known_deltas.md);
    #   * a3f158bef (2026-09-13) the substep count is checked before it is
    #     converted to a signed counter; an unrepresentable count or an
    #     invalid density/thickness stops the affected column and reports
    #     a failed call through the health path, normal schedules
    #     unchanged and no iteration cap applied
    #     (tests/test_wdm6_count_safety.py, tests/test_wdm6_count_cuda.py).
    # RE-PINNED 2026-09-30 (A144): the rain condensation cap at
    # module_mp_wdm6.F:1255 keeps the rate's sign, so rain with no number no
    # longer evaporates half the saturation deficit and feeds ice deposition
    # that took vapour below zero (both WDM6 convective cases stopped at step
    # 4).  WRF v4.6.1's Fortran goes negative on the same captured columns and
    # stays nonnegative with this one rule; both cases now run 1 h with the
    # vapour check on (tests/test_wdm6_numberless_rain.py,
    # docs/wdm6_oracle_known_deltas.md section 6).
    # Re-pinned for A146 (a98f2482e) on top: constant-divisor float
    # divisions spelled __fdiv_rn.  A144 alone was 53ac977a, A146 alone
    # 0df1f08e; previously 6f528e1b1df03047 (a3f158bef).
    # RE-PINNED 2026-10-07 (lane/verify-wdm6-mp16): the column oracle
    # tools/wdm6_wrf461_oracle grades the kernel word for word against WRF
    # v4.6.1 (tests/test_wdm6_wrf461_oracle.py); wdm6init's own float32
    # coefficient words, WOOF's own float32 exp/log/pow, the Fortran's
    # operation order (denfac, PLM tl2/th2, lamdr**3) and radii formed in
    # metres.  Previously e1c1a666.
    # RE-PINNED 2026-10-14 (lane/290-A4, correct mode WB5 in WDM6): the
    # rain-out integral gains a WDM6_PLM_SURFACE arm that books the
    # straddling cell's below-ground part with the remap's own PLM profile
    # (physics.mode = "correct", RunConfig.wdm6_conservative_surface).  With
    # the macro undefined the preprocessed unit is the previous bytes
    # (add2512f, tests/test_correct_mode_water_budget.py strips the arm and
    # compares the sha, and compares the compiled image), so exact mode is
    # the WRF v4.6.1 word-for-word kernel unchanged.
    "wdm6":
        "d5277e0fa7e2dd3cc8bd2e4b02443c2a453e18c9f5d0e002fbfc37d3bdb53743",
    # 5165b9485 chore(wdm6): the divergence gets a citation, the constants get one home
    # RE-PINNED 2026-10-07 (lane/verify-wdm6-mp16): refl10cm_wdm6 in the
    # Fortran's declared REAL/DOUBLE types with WRF's radar words, graded by
    # the same oracle; refl.cu's copied prologue is unchanged.  Previously
    # 5dff160d.
    "wdm6_refl":
        "16b57270344b0f89c12f44c5a11619b162a0a2e5ed73fe1cdb8ac547a5dfea32",
    # c5afbc870 Batch YSU output validation
    "ysu_validation":
        "ed125e770df19cb3161c4a8bed53e55cc0d740f521f318cf9166e2a1063ddd25",
    # lane/urban-ucm: the single-layer urban canopy model (sf_urban_physics=1),
    # bitwise against WRF v4.7.1's urban, lsm and noahmp_urban
    # (tests/test_urban_ucm_*wrf471_parity.py).  Moved on lane/urban-infra
    # 2026-09-30: the green-roof constants are read from __constant__
    # memory so NVRTC 12.9.86 cannot mis-fold their sums (bitwise under
    # 12.9.86 and 13.4.92 since).  Previously 6589c991.  Moved on
    # lane/urban-physics 2026-09-30: ucm_overrides blends the UCM's 2 m value
    # into Noah-MP's T2 as the absolute temperature it is, the one named
    # divergence from WRF's surface_driver.F:3393 (measured: bitwise against
    # WRF built with that line fixed, tests/test_urban_ucm_noahmp_wrf471_
    # parity.py; the 750 m Los Angeles run's city T2 had fallen 11 K below
    # its own skin at 1,900 m).  Previously a692015a.
    # Reference comments now record the original WRF v4.6.1 authority and
    # shared native twin. No executable statement changed; both oracles
    # retain exact float32 output words. Previously 30436e47.
    "urban_ucm":
        "ee5bb8e48b297b0d5304047b7d764103a5c22bc1fb307df13670c122c39e588b",
    # a28017016 feat(urban-bep): MYJURB (MYJ under BEP/BEP+BEM), word-identical
    # to WRF v4.7.1 (tests/test_myjurb_wrf471_parity.py)
    # Re-pinned for A146 on lane/281-nvrtc-literal-div: the flag_bep lower
    # TKE correction's division by 11.788 (module_bl_myjurb.F:426-429) is
    # spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by the
    # rounded reciprocal for sm_120 under -ftz=true.  sm_89 compiles the
    # same division either way.  Previously a25a77f1.
    "myjurb":
        "ff8c000f5fdbe9f43f903a00ef9a20c760635c5321e2912257ebfe2d172a26e8",
    # lane/282-terrain-drag: sub-grid terrain drag from WRF v4.7.1, the
    # topo_wind static coefficients (start_em.F:1539-1626), gwd_opt = 1
    # (module_bl_gwdo.F -> bl_gwdo.F90) and gwd_opt = 3 (module_bl_gwdo_gsl.F),
    # bit for bit against WRF's own Fortran on both cards
    # (tests/test_terrain_drag_wrf471_parity.py).  Compiled by its own direct
    # NVRTC loader (woof/core/terrain_drag.py), -fmad=false --ftz=false.
    # Re-pinned for the SASE upper-bracket PBL-top helper.  The complete
    # module, its bit oracles, helper and mutation controls pass on
    # sm_89 and sm_120; the WRF scheme arithmetic is unchanged.
    # Re-pinned by lane/verify-gsl-drag: gwdo_gsl_column takes the SPP PBL
    # pattern (rstoch = pattern_spp_pbl(i,kts,j) under spp_pbl = 1, null
    # otherwise), as WRF v4.6.1's gwdo_gsl does.  The WRF v4.6.1 column
    # oracle (tests/test_gsl_drag_wrf461_column_oracle.py, 20 cases, four
    # with SPP) is 0 ULP on every lane on sm_120, strict and default.
    "terrain_drag":
        '5da35b596b0843a087353795adacb5cceb9a6f5f4ce747745405536551182218',

    # ebcf4edd2 lane/urban-bep: the BEP column (sf_urban_physics=2)
    # (tests/test_urban_bep_wrf471_parity.py)
    # Re-pinned golden/wrf-fixes: the member-offset static_assert guard asks
    # for __builtin_offsetof itself instead of NVRTC >= 13 (NVRTC 13.2 lacks
    # it and could not compile the unit); PTX byte-identical before and after
    # on NVRTC 13.4 at compute_90, compute_100 and compute_120.
    "urban_bep":
        "255ab9f79dd10c3bb68c03997e91a361a58ad18695db38071e3a91412352a6c6",
    # 04118fa24 fix(urban-bem): BEP+BEM (sf_urban_physics=3) bit-identical on
    # sm_89 and sm_120; compiled through woof/core/urban_bem.py's own unit
    # (tests/test_urban_bem_wrf471_parity.py)
    "urban_bep_bem":
        "1b84b9d2a7309de203f311a28e6d8db79ba4708800747123fe664ec8014107c5",
    # 466fce443 lane/urban-bep: the Noah/Noah-MP BEP surface couple
    # (tests/test_urban_bep_couple_wrf471_parity.py); re-pinned by
    # lane/verify-urban-bep: the calm-column guard (WRF divides the rural
    # stress by a zero level-1 speed; tests/test_urban_bep_wrf461_column_oracle.py)
    "urban_bep_couple":
        "f9063118af4f7c7acefa4468e75354638125ea4c083c18904b8d15049d2ede0f",
    # The translation units the 2026-09-30 speed lanes added (eleven; the
    # megakernel lane's mynn_seaice_glue, phy_column and phy_glue left the
    # tree with its revert, A147, and their pins with them).  Each
    # lane shipped its .cu with no pin, which is what this file's reverse
    # gate reported at the six-lane merge 74a4374af; they are pinned at the
    # bytes that merge carries, in the sweep that found them rather than in
    # the commits that added them.  Every lane recorded its change as
    # byte-identical output against d6929cb8d.
    #
    # lane/speed-dycore-host, 66af318bc perf(dycore): the RK time-t copies
    # and tendency clears run as one word-copy and one word-clear launch,
    # and surface w, face mass and held heating each run as one point-local
    # launch (tests/test_rk_bookkeeping.py, test_surface_w_fused.py,
    # test_face_mass_fused.py, test_held_heating_fused.py).
    "face_mass":
        "e68a95a475c3b2b958cf0e40805537bac5c0c075600b2b2ef81dae4261010ec0",
    "held_heating":
        "9228ead79c478bf4c8c7dd15dec9b917dcbd6a46cff0478dee51c97c25881c89",
    # 2.8.2: four words per thread, aligned uint4 transactions, and scalar
    # fallbacks for slices and tails. test_bandwidth_word_kernels.py checks
    # copy payloads and zero words including NaNs and alignment boundaries.
    # An eight-step default-suite run retains all 154 canonical arrays and
    # byte-identical Rust history. Word moves have no WRF Fortran oracle.
    "rk_bookkeeping":
        "53ecc9f20a4e8a18f23a6ed17e43decdb80e2083660e90771a4f3d18b5dd8e66",
    # 2.8.2: vector add/theta glue and fused theta-forcing export retain
    # every eager FP32 arithmetic boundary and operand order. The tests in
    # test_bandwidth_glue.py compare exact CuPy words for physical values,
    # NaNs, subnormals, division extremes, broadcasts and alias fallbacks.
    # The eight-step default-suite state/history proof covers the adds and
    # theta construction. test_dycore_advective_forcing_export.py exercises
    # the export through a real Grell-Freitas step. These array operators
    # have no dedicated WRF Fortran oracle.
    "bandwidth_glue":
        "8ca05bd9653f73aa06b90ab25bfaf10468517b410f01d09725cc5de376458f12",
    # WRF module_bc_em.F clamps outside terrain donors on nonperiodic
    # edges. The former inside-slope copies doubled the normal component.
    # test_surface_w_fused.py checks both launch paths against the WRF
    # clamped-index formula, including periodic controls and upper W.
    "surface_w":
        'b47cecec63e457c155bdcc4a1a2c9ed299f0a5352c52e5af4f7814671d2526f8',
    # lane/speed-rrtmg-legacy.  rrtmg_legacy_adapter: added at 2c64326aa
    # (the adapter keeps the radiation call on the device), moved by
    # cfd6503b7 and last by 378191e61 (results, ozone and radius conversion
    # on the device).  rrtmg_legacy_prep and rrtmg_lw_chain_coalesced:
    # 668fa4c56 (the wrapper prep on the device, the coalesced g-point
    # slabs).  rrtmg_lw_zbatched: added at bfc177208 (one-launch taumol),
    # moved by a44612163 and last by 77ae92552 (a summing thread per
    # (column, level) row).  Exercised by
    # tests/test_rrtmg_legacy_device_glue.py, test_rrtmg_legacy_prep_device.py
    # and test_rrtmg_lw_batched_layout.py.  Moved by lane/286-fork-mynn:
    # rla_mynn_gsd41 added (the GSD MYNN v4.1 subgrid-cloud merge, selected
    # only under bl_mynn_version = "gsd_41"); every existing entry point is
    # unchanged text.  Previously b19b4e13.
    # CONUS strict wrapper repair: WRF driver SW round trip, 0/56523227 words
    # at each of steps 67 and 100, receipts in conus-suite-exact/mp-rad-lsm.
    "rrtmg_legacy_adapter":
        "50f9f763628cd2253192fa2a82e1379c2abd59c48ecea6cbaf5d9e279c960717",
    # Coastal LW native-entry regression: the positivity guard counts only
    # retained interfaces; WRF overwrites the final interface with zero.
    # Re-pinned for the source-selected NOAA WRF 3.9 SW cloud optics:
    # land/water fallback radii 5.4/9.6 microns and snow fraction cap 1.0.
    # The WRF 4.6.1 default stays behind RP_CLOUD_FORM=0. Both forms were
    # exercised by test_rrtmg_source_cloud_form_gpu.py in the node4 gate;
    # the measured node4 source-transition receipt records these raw bytes.
    # Re-pinned on golden/ports-rad: one kernel APPENDED after the last
    # existing one, rp_cldfra2 (WRF's binary cloud fraction, icloud = 2,
    # module_radiation_driver.F cal_cldfra2); every earlier line of the unit
    # is byte-identical (the diff against 2.8.9 is 13 added lines at EOF).
    # Measurement (G1, RTX PRO 6000 Blackwell, 2026-10-12, tip 9e9b50f03 vs
    # cd336e62d6): real CLI legacy RRTMG runs that do not select icloud = 2,
    # 180 s, default and strict arithmetic -- all 300 history field-frame
    # hashes and all 144 checkpoint arrays byte-identical in each mode
    # (lanes/ports-rad/receipts/identity/cmp-legacy-{default,strict}.json);
    # the new kernel's own words are graded against WRF's cal_cldfra2 by
    # tests/test_cldfra2_gpu.py.
    "rrtmg_legacy_prep":
        # 2.9.0 round 2: golden/ports-mp's Ferrier consumer edit (024c62024)
        # plus golden/ports-rad's appended rp_cldfra2, each pinned on its lane;
        # the combined bytes are both changes and nothing else.
        "756863c073537b8bde4879f0c0b7fa1f06b8d862b3f24353c1118dc05ef08d7b",
    "rrtmg_lw_chain_coalesced":
        "2682d172388d4a31ae11be0168bdd13b33be9c1b8e77e7ca92e86a8f9a822401",
    "rrtmg_lw_zbatched":
        "5611794c5c9e44815ad00e28af7367428beda949afe205654ecb959665de7a8e",
    # lane/281-zadvect-implicit (A158): WRF 4.7.1's implicit-explicit
    # vertical advection, graded word for word against WRF's compiled
    # routines by tests/test_zadvect_implicit.py.  Moved once on the lane:
    # ieva_solve_s takes WRF's t0 as a shift, so the theta solve reads
    # theta - t0 as WRF's does (a uniform column stays uniform).  Moved
    # again by lane/281-ieva-units (A179): ieva_solve_w's two boundary
    # terms take consistent units (the lower one uncouples the u/v
    # tendencies, the upper one divides by g), a declared divergence from
    # WRF 4.7.1 graded against WRF's routine with the same corrections.
    # Moved a third time by lane/sol-hrrr-zadvect (37a99fcd4, lane tip
    # fd1772ba0, merged at dd5ded73f): zadvect_implicit_variant =
    # "wrf_legacy" selects the earlier WRF split (upwind-level winds with
    # the mass-point map factor, cx+cy), current-mass face coefficients in
    # the u/v solves and the earlier upper-w grouping (divide the whole
    # increment by g); the "wrf_471" default runs the operator above
    # unchanged.  Graded word for word against native Fortran extracted from
    # the pinned NOAA-EMC/HRRR WRF source (40ee6058c) by
    # tests/test_zadvect_legacy.py: 33,320 words at max ULP 0 for the legacy
    # arm and 4,381 at max ULP 0 for wrf_471, on an RTX 5090 at NVRTC 13.4,
    # whose reading also gives local frames of 1040/2064/4112 B at
    # IEVA_KMAX 65/129/257, unchanged, so no frame recording or price moves.
    # CONUS strict stored MF_VX_INV carried through the column-mass caller;
    # the fast branch retains its original quotient.
    "ieva":
        "c8ca76913ae424717c93b5a2dc48113f7bd628fefd013b0c105d445aa234c528",
    # Lane 281-namelist-gaps: WRF v4.7.1 slope_rad / topo_shading
    # (module_radiation_driver.F toposhad/topo_rad_adj), held bit for bit
    # to WRF's Fortran by tests/test_topo_radiation.py on a card.
    # Re-pinned by lane verify-topo-radiation (2026-10-07): the physics is
    # unchanged; SIN/COS/TAN/ASIN/ACOS/ATAN/EXP/** became the float32 libm
    # words (they were the double function rounded to float, which missed
    # WRF v4.6.1 on glibc 2.43 at 3,660 of 425,468 chain values) and ATAN2
    # became correctly rounded.  Measured on W1 (RTX PRO 6000, NVRTC
    # 12.9.86): 0 of 425,468 chain values differ from WRF v4.6.1 on glibc
    # 2.43, strict and default arithmetic.
    "topo_radiation":
        'eed766ccaa9cd8e856423c407750cffb829a9e8e01c0b28c5ef91b9d16861de8',
    # Noah mosaic (sf_surface_mosaic = 1): WRF v4.7.1 lsm_mosaic and its
    # ordinary and glacial SFLX subtrees, bitwise against the byte-unmodified
    # WRF column oracle at every non-FTZ word
    # (tests/test_noah_mosaic_wrf471_parity.py).
    "noah_mosaic":
        "8dab69ae3a7cb90ad41436b97ade1f4728e4030d3fc0b86fced12089ff21f791",
    # lane/aq-core (smoke, aerosol and air-quality program).  chem_bdy:
    # WRF-Chem's outer-domain flow_dep_bdy_chem, bit-identical to the
    # Fortran on 24 oracle cases (tests/test_chem_bdy_wrf471_parity.py).
    # chem_outputs: the table diagnostics' term programs, PM2_5_DRY/PM10 at
    # 0 ULP to sum_pm_gocart (tests/test_chem_sum_pm_wrf471_parity.py,
    # tests/test_chem_outputs.py).  chem_ledger: the mass ledger's
    # row-parallel float64 reduction, a diagnostic that moves no model word
    # (tests/test_chem_transport.py's ledger assertions).
    "chem_bdy":
        "2c8e91bd3c2af915967aa1287e946961d5e9f0b9403ab66281d7a94c5f222d67",
    "chem_outputs":
        "313503588f5c85ace60be4967995046c958e61ad63b8f1cd36b509b0348f0d99",
    # Ranked interior bounds (280def6e2f) are pinned in the extension manifest.
    # The 20-step prepared-store seam gate matches resident state/history
    # words and float64 ledger totals to the declared reduction tolerance.
    # chem_prep: chem_prep's column inputs, 0 ULP to the Fortran on the CPU
    # reference and the card (tests/test_chem_prep_wrf471_parity.py).
    # chem_vertmx: vertmx with deposition as its lower boundary, 0 ULP the
    # same way (tests/test_chem_vertmx_wrf471_parity.py); only the Thomas
    # recurrences live in the thread frame.  mono_advection: WRF's
    # advect_scalar_mono, 0 ULP on 74 oracle cases
    # (tests/test_chem_advect_mono_wrf471_parity.py).
    "chem_prep":
        "bc3547b12038132555c7cb39ce93746719a51a50ad457f9e065ec002ed4ceb09",
    # Re-pinned for the HRRR v4.1.21 fork's enh_vermix (smoke/
    # dry_dep_driver.F:178-195), behind a launch flag that is 0 unless
    # RunConfig.enh_vermix: the stock floors are unchanged text, the 74
    # WRF 4.7.1 oracle cases stay 0 ULP (test_chem_vertmx_wrf471_parity),
    # and tests/test_chem_smoke_fork_processes_gpu.py holds the raised
    # ekmfull words to the fork's MAX(1.1*ekm, 10/20) rule.
    "chem_vertmx":
        "c571534c6ab5a4300301a15cbaed2a033348a3a3faf6b358b6d80a492c1a88da",
    "mono_advection":
        "ba9417639e3f0105dcf67e0e3eb18ae59d33dac00c1d0c84009a8b1c17c3a853",
    # lane/aq-smoke.  chem_fire: GSL smoke_dust's fire preparation, unit
    # conversion and diurnal injection, bitwise to the gfortran oracle on the
    # CPU reference and on sm_89 and sm_120 (tests/test_chem_fire_emissions.py).
    # chem_plumerise: the Freitas plume (WRF-Chem module_chem_plumerise_scalar
    # and GSL module_plumerise), transcribed by
    # tools/chem_wrf471_oracle/transcribe_plume.py and compiled unflushed
    # through woof.core.chem_plumerise_cache; bitwise on both oracles
    # (tests/test_chem_plumerise_wrf471_parity.py,
    # tests/test_chem_plumerise_frp_parity.py).  chem_wetdep_ls: WRF-Chem's
    # wetdep_ls, bitwise (tests/test_chem_wetdep_ls_wrf471_parity.py).
    "chem_fire":
        "fed467fff8deff3590bd2323b9d1eadd4246ec4b5486f97b407cb90053ebb55d",
    "chem_plumerise":
        "e429393ea14973998248762040278b06e886a5de4deeb6c118716f7b27a0e798",
    "chem_wetdep_ls":
        "cc7e80c2205ff72b7e82999ec13bfc7b95240720e83f1b9d7369da5574f4ba2e",
    # lane/aq-gocart (GOCART-lite).  Each is a CUDA port of WRF-Chem v4.7.1
    # proven against gfortran -O0 builds of the pinned WRF-Chem sources
    # (tools/chem_wrf471_oracle/gocart/), with the parity test named:
    # chem_mp_coupling (69b71f60c): GSL's aerosol-aware Thompson NIFA/NWFA
    #   diagnosis, 0 ULP (test_chem_mp_coupling.py).
    # chem_inventory (9d14eb690): emissions_driver's emiss_opt=6 block,
    #   0 ULP (test_chem_inventory_wrf471_parity.py).
    # chem_settling and chem_drydep_gocart (019a96c4c): gocart_settling and
    #   gocart_drydep_driver, 0 ULP (test_chem_settling_wrf471_parity.py,
    #   test_chem_drydep_wrf471_parity.py).
    # chem_dust and chem_seasalt (0c7e211f7): GOCART dust (dust_opt=1),
    #   AFWA dust (dust_opt=3) and GOCART sea salt, 0 ULP
    #   (test_chem_dust_wrf471_parity.py, test_chem_afwa_wrf471_parity.py,
    #   test_chem_seasalt_wrf471_parity.py).
    # chem_rrtmgp_aerosol (653596a4a): RTE-RRTMGP's MERRA aerosol optics
    #   lookup, 0 ULP (test_chem_rrtmgp_aerosol_parity.py).
    # chem_sulfur and chem_ageing (a812bfb79): gocart_chem's SO2/DMS
    #   oxidation and gocart_aerosols' carbon aging, 0 ULP except MSA's
    #   1 ULP and the solar geometry's documented trig-stub residual
    #   (test_chem_sulfur_wrf471_parity.py, test_chem_ageing_wrf471_parity.py).
    #   chem_sulfur re-pinned on lane/verify-chem-aq-smoke: szangle's SIN/COS
    #   moved from CUDA's sinf/cosf to WOOF's own float32 trig, which retires
    #   the trig-stub residual (cos(SZA) up to 36 ULP, tcosz up to 25 ULP,
    #   SO2/SO4 1 ULP); every sulfur output now grades 0 ULP in native and
    #   oracle-solar modes on all 48 cases (test_chem_sulfur_wrf471_parity.py).
    "chem_mp_coupling":
        "d7f9c5727eec775045c791f0d8922811eafaf2e6dc8540527a7ea3e42c847ed2",
    "chem_inventory":
        "3cf17a7d466128c3a7a6ad206c10c967ab8b567b1f60758f19f98a7644039aeb",
    "chem_settling":
        "7c50c6ed45822901323c43dd62453157e7fc7c52c2279d1b409950d041d5efce",
    "chem_drydep_gocart":
        "dd82e5a1dcb8ee327785c185842fa4a435c19d313f7d7353ce86ebf554f789e3",
    "chem_dust":
        "250bcf60edc1afbeb64ee621a1d3e51f8695acddfb0bcb110f980a7521c96b70",
    "chem_seasalt":
        "394e116c698293482f7e96ce47018be175506d17b8b397729e80f8db318d55c3",
    "chem_rrtmgp_aerosol":
        "137d81d2d70e2fcd6f5e83c11f1bcab0ee63a1a7dc880c50cca4e7d926647f72",
    "chem_sulfur":
        "f01a5d1a17dc717b22acd0f06b612d4d2a32f92276ad252b3d63640bbe51a2d5",
    "chem_ageing":
        "895cc8c46773b5354ed05cbcd1a703babdba536fc54f0d2dc2bea5c578f33be0",
    # chem_optics: module_optical_averaging's GOCART volume optics, WRF's own
    #   Mie fits, aer_opt_out's EXTCOF55 and its column integral, and RRTMG
    #   SW's aerosol band conversion, 0 ULP and zero word mismatches on 33
    #   cases plus the RRTMG edge column; each output store optional
    #   (test_chem_optics_wrf471_parity.py).
    "chem_optics":
        "6b832bcd076af6b67c96a019d5f1db00206f6d75398f2aff2dc5b2b61263da87",
    # lane/aq-cams.  chem_drydep_wesely: WRF-Chem v4.7.1's Wesely gas dry
    # deposition (rc, landusevg, depvel, cellvg), 0 ULP to the unmodified
    # Fortran on 11 cases / 4,752 columns on sm_89 and sm_120
    # (tests/test_chem_wesely_wrf471_parity.py).
    # Constant denominators use wd_div and the existing WFloat div.rn.f32
    # operator. Passing WFloat to the float-only __fdiv_rn cannot compile.
    # The native oracle cases grade these repaired bytes on the card.
    "chem_drydep_wesely":
        "33761f22276d75382a03dc2eb67ba1d866e24540eabaedc20daa5ebe31cda63d",
    # Lane 286-aer-swint: WRF swint_opt = 1 (radconst/calc_coszen at the
    # current xtime, update_swinterp_parameters, interp_sw_radiation).
    # Re-pinned by lane verify-swint-opt (2026-10-07): the kernels follow
    # WRF v4.6.1 (exponent-zero ratio rule, WRF cc02f989f; [-1, 1] cosine
    # clamp, WRF 650be0c7d), WOOF's own float32 libm through
    # glibc_flt32.cuh and glibc_trig_flt32.cuh, held word for word to
    # v4.6.1's gfortran -O0 Fortran (tools/swint_wrf461_oracle) and to the
    # HRRR fork's outside those two fixes by tests/test_swint_interpolation.py
    # on a card.
    # Lane 286-aer-swint: WRF aer_opt = 3 shortwave optics (gt_aod and
    # calc_aerosol_rrtmg_sw of the NOAA-EMC/HRRR v4.1.21 fork, RSLF,
    # Bolton RH, the rural Lagrange tables), glibc expf through
    # glibc_flt32.cuh, held word for word to the fork's gfortran/glibc
    # Fortran by tests/test_rrtmg_aerosol_optics.py on a card.
    # Re-pinned for prescribed smoke: source-order 4.0 mass extinction,
    # layer cap 3.0 and no RH enhancement. The pinned smoke_aer3_oracle
    # fixture and node4 smoke optical/provider gates grade these bytes;
    # zero smoke remains an exact control. No forecast skill is inferred.
    # Re-pinned for the NOAA MPAS land/water aerosol type (RRFS hex gap 2):
    # maritime rows and a separate rrtmg_aer3_typed_sw_optics entry point;
    # the rural and smoke kernels are unchanged text, and
    # tests/test_rrtmg_aer3_landmask_type.py holds the new kernel to the
    # NOAA MPAS fork's Fortran word for word on a mixed deck.
    "rrtmg_aer3":
        "0ab8dab99a9b49c93699c0e3ce6778d63a6e8e1eb39966a1db49d4847db10dd3",
    # Explicit prescribed profiles only: device finite/unit guards,
    # bounded UTC interpolation and congruent-grid slab gathering.
    # The node4 8-test adapter/provider gate binds this exact raw source.
    "rrtmg_smoke_manifest":
        "ac810c4ffc873f084882adb695eccdbf84d8c7d28af1654c3d01755770460ece",
    "swint":
        "9a79b7e43cb92553ae454975a943acedc3d8510b5c7eae59ec2c5bb8c8b47c5b",
    # cu_physics = 6, WRF v4.6.1's original Tiedtke (module_cu_tiedtke.F) as
    # one column kernel; tools/tiedtke_wrf461_oracle/check.py --gpu holds
    # it word for word to the Fortran in default and strict arithmetic
    # (tests/test_tiedtke_cu6_gpu.py).  PINNED at 2.8.9 when it joined.
    # Re-pinned 2026-10-29: the vector_row argument selects F:588's libmvec
    # lanes of WRF's stock -O2 build (WOOF_WRF_LIBMVEC=1); at 0, the
    # default, every word is the earlier kernel's.  A captured WRF 4.6.1
    # step (499 x 499 x 44) replays at 0 of 66,483,267 words with it
    # (tests/test_tiedtke_cu6_stock_capture_gpu.py).
    # CONUS strict glibc 2.39 tangent: 0/66483267 captured words, step 100.
    # The vectorized glibc 2.43 path keeps its tangent and exp selection.
    "tiedtke":
        "322b181169e1192f4df5c6ad17fd7d463b3eb7893c7362474aa634efc40bd25e",
    # WRF 4.6.1 and 4.8.0 solar words are checked independently on a card.
    "rrtmg_legacy_solar":
        "d0d9fa08a5ff854f9059f0151ddaa2fdb80ca1a13245c774b33432a065bb5d64",
    # WRF 4.8 YSU (ysu_version = "wrf_480"): graded against WRF 4.8 MMM
    # 550b5b4 bl_ysu.F90 by tools/ysu_generations_oracle/gate.py.
    # Re-pinned for the YSU column speed change (ysu_480): register-carried and
    # index-only changes, no floating operation moved.  Every output word
    # of every launch in one captured model step of a 300x300x49 3 km real
    # case equals the previous kernel's (default and strict builds, sm_89
    # and sm_120); first-hour wrfout byte-identical.  Previously 74892789; after the
    # fold's entrainment Courant cap (1edfbfe953) previously 44acadb4.
    "ysu_480":
        "3692a061665fb01fe1b900d4f63ed5cd20b76eb90ac375d4977775d432d74ddc",
    # The 2.9.1 intake of the WRF exact chain (r290/exact-chain 6cb618fce):
    # source identity here, numerical gates in their own suites.
    # physics_exner, thompson_classic and thompson_fork are the strict
    # (GPUWM_WRF_EXACT=1) Exner, classic-Thompson and fork-Thompson units;
    # the *_default units are what a default process compiles for
    # nssl2*, rrtmgp_* and ysu (ysu_default.cu is 2.9.1's ysu.cu, byte for
    # byte), kernels/__init__.py _DEFAULT_SOURCE_MODULES.
    "nssl2_diagnostics_default": "a95ae9e0bc3dd20a13865cfa6d1148d2a78ee5d7c17c9c1bca9a0c8dbdf19868",
    "nssl2_driver_support_default": "12f193c6918056b995577930304867cd8369a46ab1facfcde589d244cbd6eca8",
    "nssl2_fused_gs_default": "07dff4bf34509e89b75683a83d356c4691b6eab1aa96140af3e6fa5b00fe429b",
    "nssl2_nucond_default": "b4819a1e912924fe2e30c33897585e3a46c32e89dcb3ab43eb1ee32b212b6db1",
    "physics_exner": "14d8847f39847e9637efc83b63dbf323eddfe4d3f21675210b6398ffe85c3636",
    "rrtmgp_gas_default": "d13b914a28ad8b7ef88904977c7e33b9a7d7e800592473c0fb8debebca031edf",
    "rrtmgp_rte_default": "731bd84717398320a1710b2c151a466b9e5bf1a30b633d858fd31d7162dbd173",
    "thompson_classic": "c4b7cb844354e29782f9d6c0b2d97267bb90d08e371e90dde90f53342566130e",
    "thompson_fork": "c7313ce62d863bdeb9478b7a503c8faffae6d17c8a20cbb1b6d458c216e4c947",
    "ysu_default": "f56e3ec7646a111d07c5447383ae136b2959b5f4545441a598b51be16ccdc768",
    # Fitch wind-farm drag and turbine TKE inside the GPU MYNN step (labs/windfarm 73cb37dd11, 2.9.11; opt-in
    # --wind-farms / WRF windfarm_opt = 1): its translation unit is pinned in the commit that ships it.
    "wind_farm": "693cd368feb1beffdc8805017806b13fe95a9558fb3a43e2e3a657dd89100412",
}

_FROZEN = _frozen_module()
# WRF 4.7 Noah-MP night snow aging (noahmp_snow_age = "wrf_470"): the
# radiation unit gains the night-aging entry beside the unchanged daylight
# one; graded against WRF 4.7.0 and 4.6.1 full columns by
# tests/test_noahmp_snow_age_gpu.py (19,614 words, both generations, both
# arithmetic modes).  Its new digest f7317dcf... is pinned in
# FROZEN_MODULE_DIGESTS itself (tests/test_mp8_frozen.py), so it is not a
# drift entry here.
PINNED: dict[str, str] = dict(
    (name, digests[0]) for name, digests in _FROZEN.FROZEN_MODULE_DIGESTS.items()
    if name != "ysu")
# The unreachable strict ysu.cu copy is retired; ysu_default and ysu_480
# retain independent source pins below. Historical freeze evidence stays.
PINNED.update((name, sha) for name, (sha, _c) in RE_PINNED_DRIFT.items())
PINNED.update(BASELINE_PINNED)
_EXTENSION_PINS = _extension_pins()
_merge_extension_pins(PINNED, _EXTENSION_PINS["modules"])

# These are the complete headers the real loader prepends. common.cuh
# retains its existing authority; the other rows close the former *.cu-only
# gap without changing any source bytes. The GF gamma numerical evidence is
# docs/gf_gamma_known_delta.md and tests/test_gf_gamma_correctly_rounded.py;
# header assembly is independently checked by test_kernel_loader_inert.py.
PINNED_HEADERS = {
    # Existing fork radar transcription, reused unchanged by the initial helper.
    "thompson_fork.cu": "c7313ce62d863bdeb9478b7a503c8faffae6d17c8a20cbb1b6d458c216e4c947",

    # New Goddard band and sounding tables (lane ra5-goddard), prepended to
    # the goddard unit only (kernels._EXTRA_HEADERS).
    # 2.9.0 round 2: eight constant divisions spelled g_div (A146), whose
    # overloads are __fdiv_rn for a float operand and WRF's double division
    # for an fp_kind one (a bare __fdiv_rn had narrowed the fp_kind operands
    # to float and made g_pow(float, GReal) ambiguous on NVRTC 13.4);
    # emitted by translate_bands.py, regenerated from WRF v4.6.1 (d66e442)
    # source on test host A, byte-identical to this file.
    "goddard_bands.cuh": "ea0a43f2449963f12b97b01638c023825129acf97dd8825d3a4efcb704cd21b0",
    "goddard_sounding.cuh": "1e3b89eb9e87acc98ebba5e55d81facffd5d47d58db82ca2743dca02e4d75fe6",
    "thompson_column_support.cuh": "3c1c267c853d22fe67c327a105145d6e2200000209fe30dbfef353b70ba86b4f",
    # Re-pinned at 2.8.9: the blank line after the closing brace at EOF
    # (ddda6dabc) is gone; the 2.8.9 cut's export whitespace check refused it.
    "thompson_column_radar.cuh": "30658defe41421cade975ef6456adaf0540bf13ad09d3cdea6227573c6e623f5",
    # Re-pinned at 2.8.9: translate.py no longer emits a blank after an
    # if-then brace, so 187 trailing blanks left the regenerated header (the
    # 2.8.9 cut's export whitespace check refused them). Only line-end bytes
    # changed; every statement and table literal is the same.
    "thompson_column_generated.cuh": "f07dd881fbec30952ba3d17b16012693e4ea689f50dcefde350c0222473fb552",

    "urban_ucm480_support.cuh": "6b5db2885a7e00d5c8ec65f7e3857efc9cf30c2264c9e5512020373e891089d6",
    "urban_ucm480.cuh": "bcb062c9ad2854e54ae2205ad898f26aa6153a8144729491577a631ebcc5cfaa",
    "urban_ucm.cu": "ee5bb8e48b297b0d5304047b7d764103a5c22bc1fb307df13670c122c39e588b",
    # Lake lower bounds, REAL kinds and division rounding are part of the
    # column contract; test_lake_contract regenerates the WRF statements.
    # ABS sign clearing includes negative zero, subnormals and signed NaNs;
    # native Fortran word controls and the CUDA probe hold that boundary.
    # RE-PINNED at 2.8.5 from 2c67731aae313025 and f31a4a7144c95ad3 by the
    # notice correction: one comment line in each header (line 2, line 3)
    # named tools/lake_wrf461_oracle/NOTICE.txt, which neither distribution
    # carries, and promised CLM notices that file never held.  Both now name
    # licenses/LICENSE-WRF-public-domain.txt.  Comments only, measured: every
    # other line is byte-identical and the headers keep 92 and 4128 lines.
    # RE-PINNED at 2.8.6 from 5a8fb446ae3b0584 by the lake arena (A3/A13):
    # on the device a LakeStorage takes its elements from a per-launch
    # global arena (strided, coalesced) instead of the 14,400 B local frame
    # CUDA backed for every resident thread of the card; the host build
    # keeps inline storage.  Addressing only; no arithmetic changed.
    # RE-PINNED at lane/verify-clm-lake from 0c931e51e4e449b6 and
    # 196340026552b827 by the correctly rounded lake intrinsics: the
    # transcription's binary64 exp, log, log10, sin and atan now read
    # lake_exp, ... (translate.py), lake_support.cuh defines those and the
    # binary64 power through f64cr.cuh, and REAL(4) calls of them no longer
    # compile.  Measured: the CUDA lake equals native WRF linked to
    # correctly rounded functions in every state and output word of the
    # 112-column regime campaign (tools/lake_wrf461_oracle/regimes.py), where
    # CUDA's own library left hfx and grdflx in 1,223 words.
    "lake_support.cuh": "c21f1cf9deb1930e9ad5e24b48247cedf8de8832fdb92e97dfbf938122ccb106",
    "lake_wrf.cuh": "91ccdd68fc9b87564a28f1554faa2a913171e02d963d86b46be292c65e47fffd",
    # WOOF's correctly rounded binary64 exp, log, log10, sin, atan, pow for
    # the lake; tests/test_lake_f64cr.py grades it on the device against
    # 240-bit mpmath values and the constant block against its generator.
    # gp-libm64: new Rust libm 0.2.16 and glibc 2.39 log1pf twins.
    "portable_libm64.cuh": "bca62ac1366a4602b0c5bd0b11a11c9ee226a1e1a1f690060c924ef94a64655e",
    # NumPy NaN payloads and explicit rounding for the real_init units.
    "real_init_common.cuh": "f4b6187b4614daa458f96689036a3bf71c12b66b0c21f5844c579c75916deca2",
    "common.cuh": _FROZEN.COMMON_CUH_SHA256,
    # noah_init reuses the full Noah unit through the real loader. Its
    # header authority is the same source pin as the forecast unit.
    "noah.cu": PINNED["noah"],
    # The order-five eta entry borrows the existing scalar-flux helpers.
    "pd_advection.cu": PINNED["pd_advection"],
    # RE-PINNED at 2.7.6 from 794c7d4123bb0642 by the notice correction: the
    # comments at lines 235-237 and 344-353 stop describing the earlier gamma
    # as derived from glibc.  Comments only, measured: with comments removed
    # the header is byte-identical, and it keeps its 552 lines.
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    "glibc_flt32.cuh": "6a0facc2047f04b071a0f77847f8d9c9079c0af5350e32a972406a81661ddc46",
    # lane/verify-milbrandt2mom-mp9: WOOF's own float32 log10 for the
    # Milbrandt-Yau units, swept against the oracle box's LOG10 over every
    # positive finite float (tools/milbrandt2_wrf461_oracle/sweep_log10.py).
    "log10_flt32.cuh": "f0635036ea7a0514279c82a8c34e2998d2b65a4e2d4f4933765ea00b259966a4",
    "rrtmgp_planck_common.cuh": "54e56d660392c533e1d89145b67bfb11f6891221a4ca8905eec1a644089a84f8",
    # A146 (a98f2482e): __fdiv_rn spellings; previously 07f5c144.
    # RE-PINNED 2026-10-03 (lane/286-fork-thompson): the WRF 3.9 fork's
    # helpers and constants appended (thompson_aa_wrf39_*), referenced only
    # by THOMPSON_AA_WRF39 arms; previously 94876bfbc38db9c7.
    # RE-PINNED 2026-10-05 by the mp=28 accumulator rework
    # (MP28_ACCUMULATOR_REWORK below): the tendency-form rain and ice
    # balance helpers the source kernels share
    # (thompson_aa_rain_balance_tendency, thompson_aa_ice_balance_tendency);
    # previously dce2673c68a7a0f8.
    "thompson_aerosol_common.cuh": 'a24a96dab64e0688106dbdda5fe518ea9390d826c1f32169126a4d3893b38e9d',
    # 399b1c017: glibc 2.43 float32 trig (Arm sinf/cosf, CORE-MATH tanf/
    # asinf/acosf/atanf) for the urban BEP column, generated and proven by
    # tools/glibc_trig_flt32_proof/.
    "glibc_trig_flt32.cuh": "bbdb54c85d361d208ea9b1a7cb49a33132026d17694ad0f42c8b0bf5460baaf0",
    # The CCPP sa-TKE-EDMF unit's prepended headers (_EXTRA_HEADERS["satmedmf"]):
    # GFS physical constants, the rounding vocabulary and the column transcribed
    # from satmedmfvdifq.F by tools/satmedmf_ccpp_oracle/transcribe.py, graded
    # with satmedmf.cu against the pinned upstream column at 0 ULP.
    "gfs_physcons.cuh": "fcdecf5a266c9f84b364599e6d6025be48f2b95f664f0b915cd1688e9ac3c461",
    "satmedmf_common.cuh": "e40ec024b95912770d90f929a3304c88834bd28c5d7075171c41ddca6fd9832f",
    "satmedmf_column.cuh": "67cbf35ff3d5f0f5f5498b98c5f38a81addcccff9afc0c28b2b2dbefd0faffbf",
    # cu6's prepended headers (_EXTRA_HEADERS["tiedtke"]): the rounded
    # arithmetic and column arena, and the routines transcribed from
    # module_cu_tiedtke.F by tools/tiedtke_wrf461_oracle/transcribe.py,
    # graded with tiedtke.cu.  PINNED at 2.8.9.
    # Re-pinned 2026-10-29 with tiedtke.cu: glibc 2.43's libmvec expf and the
    # transcribed tlucua lane twin that F:588 calls.
    "tiedtke_support.cuh": "ee7482d8f7909bbf00b3ef34054387595bdb4aa63eac2acf9c38a322f9a4c98f",
    "tiedtke_routines.cuh": "36b6b612d8a089079b7686c864474b353f208b3b174c68a9e6ad09e225bdedc1",
    # The WRF 4.8 YSU unit's own topo arm (bounded get_pblh searches).
    "ysu_480_topo.cuh": 'c1a4a8539d4d9d8301cc3edaf92115f480f76ae02b626d4248d2f33daf631b7e',
    # The UW PBL's prepended headers, in the loader's order
    # (woof/core/kernels/__init__.py _EXTRA_HEADERS["uwpbl"]): the
    # binary64 libm (tools/uwpbl_wrf471_oracle/libm64 proves it), the
    # rounding-pinned vocabulary and the CAM routines, graded with uwpbl.cu.
    "glibc_flt64.cuh": "a14aff39d4acc74d9c782726d17d0bf7ef726c40235ddeb86d6a1a0bd7c4fcb8",
    # The fork Thompson column unit's constants, generated from the fork's
    # own thompson_init/radar_init words (tools/thompson_fork_oracle/
    # make_constants_header.py; tests/test_thompson_fork_column.py holds it
    # to the committed dump).
    "uwpbl_common.cuh": "93c768b4b0cf9ddb7dfe6f0259b8bd60c7ccb0ad768aca30c2535514fcb1cdc8",
    "uwpbl_wvsat.cuh": "b78b74eeebdb3db76d5518f1f44b3b034f489cc8ff5bc3e35ae937863d8ecfaa",
    "uwpbl_vdiff.cuh": "cb580058c0213365aebe99f75306ef43ab4b05ab53922775bd66bd5ba99535c7",
    "uwpbl_zisocl.cuh": "d2ff60bed748bb8211412e36fb7ab623cea6b220b7ab5999e5565505c5074275",
    "uwpbl_caleddy.cuh": "5581569d8b48d8d17a86a45fed7930e2505f75b7526f68055e43a84ce2a6de63",
    "uwpbl_eddy.cuh": "5377dc3a6d607441652542c2af6e00e2f69b52699178b395584813b74a5f02c4",
    # RE-PINNED 2026-10-07 (lane/verify-uw-pbl-refute): a calm level-1 wind
    # (uMean == 0) now carries zero surface stress instead of WRF's NaN
    # (tests/test_uwpbl_calm_wind.py); no column with wind moves.
    "uwpbl_driver.cuh": "534ed491edf2320d6223acb85ff26bf5b1728d27e5dd1793f9506d20244fb9a1",
    # The fused RUC translation unit's own sources (woof.core.ruc_tier,
    # RUC_FUSED_SOURCES), appended after ruc.cu rather than prepended.
    # PINNED at 2.8.1 by lane speed-ruc, which added them: the fused RUC call
    # whose every output word equals the array orchestration's.  sfctmp's is
    # generated by tools/ruc_fused/gen_sfctmp.py, the driver's aliases by
    # tools/ruc_fused/build_driver.py; regenerate, prove identity, re-pin.
    # 2.8.6: regenerated for the snow lineage (ruc_snow): the dispatch's
    # three wrf_45 arms (melt-out reset before the recombination, melt
    # bookkeeping scaled by cover, SNOWFALLAC net of melt) and the v4.6.1
    # cover rebuild select on GPUWM_RUC_SNOW_V461; tests/test_ruc_sfctmp_fused.py
    # grades it against the array orchestration under both names.
    # Prevents either lineage changing unseen.  Previously 97f28c5d.
    "ruc_fused_sfctmp.cuh": "767102b293783e84887b0c5ffec1e7409a35cf014f61f157a6a9b5c057126d2f",
    # 2.8.4: mosaic mixtures and irrigation match WRF column words; mixed
    # 5,917-column calls match the resident reference for all selector pairs.
    # Prevent omitting irrigation or bypassing a lake when its model is off.
    # Forward merge of RUC selectors with the staged vegetation seam:
    # the prologue carries qvg_air, prescribed-LAI rdlai2d and the sea-ice
    # threshold together; the epilogue carries land category count,
    # irrigation and screen diagnostic selectors plus that threshold.
    # The generated header follows build_driver.py and its launcher ABI.
    # Generic forms remain irrigation/snow wrf_461, QVG wrf and diagnostic
    # flux. Named source forms are explicit. The full RUC GPU deck and
    # test_generic_ruc_defaults_match_named_legacy_after_every_gpu_call
    # must verify the merged dispatch before this tip is handed over.
    "ruc_fused_driver.cuh": 'c14efde1f4f0b5f43d6fe7b3907f394687d97c6e0b250885d819456f4fdf1d5c',
    # Selected Group A loader header from the delivered 58-sfclay-noftz-mynn-mm5-eta
    # source series; retain its exact original byte identity.
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'flt32_expf_fma.cuh': '7f41c3b20a48862aedb872fb7426d405123c599525c464a9411f21183d774aa9',
    # Selected Group A loader header from the delivered 58-sfclay-noftz-mynn-mm5-eta
    # source series; retain its exact original byte identity.
    'mynn_libm.cuh': 'bf5831efbdf1ddc1111e8f0bafc2e272b5597d7d2b12d8320675b8a938618fbb',
    # Selected Group A loader header from the delivered 58-sfclay-noftz-mynn-mm5-eta
    # source series.  Re-pinned at 2.8.8 (gfix/288-gpu-shards): ISFFLX=0 now
    # keeps the incoming CHS/CHS2/CQS2 words as WRF does and the 65 intake's
    # launch_sfclay contract states (it wrote zeros).
    'sfclay_classic.cuh': '9b384adef3a2a6fac7e721332e02acd57e63739e81e64fe48ace630a50ce6eb2',
    # Selected Group A loader header from the delivered 58-sfclay-noftz-mynn-mm5-eta
    # source series; retain its exact original byte identity.
    'surface_subnormal.cuh': '99d0aaf80b1de9e243c11cbdcebbb16198b76bd40c8207847d9284718f8021d4',
    # Selected Group A loader header from the delivered mp28-gate-all-157
    # source series; retain its exact original byte identity.
    'thompson_aerosol_libm.cuh': 'c22ea2f8b096e402685df8acf31c775a6f134af26be0646d9380e93e2df10ac0',
    # r290 exact-chain intake: source identity, numerical gates separate.
    'atan_scalar_flt32.cuh': '452366ab8ec8512b57430851ab231d494a057d49d27943cc2db12c62c7d05648',
    'glibc_flt32_extra.cuh': 'aaec88d638b359b6f0a06418bc907efb70863f11d6304342fce083a176880595',
    'morrison_constants.cuh': '23c88a2e00f6002187c62550162760c4deebb5a8eabbbda6c2f29c605a91ca37',
    'sfire_libm.cuh': '22f44144a81d60fbbd767cfcb33d260e4d1f08fdc759d962165bae70bb603ee3',
    'thompson_classic_constants.cuh': '3d0855fa3de4ca6c6a7c4afe4fcb18a7cd016e43c04de75f03c14d02095cfeea',
    # r290 exact-chain intake: source identity, numerical gates separate.
    'thompson_fork_constants.cuh': 'd13693dc55fbb263289b30154bcd6fbdec86b8cb67bd3c393750ee3b968f19c9',
    # r290 exact-chain intake: source identity, numerical gates separate.
    'gf_wrf_normalization.cuh': '2f258d559bda0b6c97a7220f2eb4c27473817ad25daf69769e858d63a1731942',
    'rrtmgp_libm.cuh': '71c883fed0cffeed7ae49cd422a8519de6d1815f1c9abf1ddedd9e5039204c96',
    # r290 exact-chain intake: source identity, numerical gates separate.
    'f64cr.cuh': '3adb76fc5f559a21a457572848772debd000f77ab6557428edfa2765f2a458bb',
    # r290 exact-chain intake: source identity, numerical gates separate.
    'nssl2_wrf_math.cuh': '15578ff62956b321b6145d1eff2bfff88a22576ff24dc4b9d4200f9ca1b7fcb0',
    # Slimming promotes the retained YSU header to the loader closure.
    'ysu_topo_default.cuh': '031efb139b827dd94fa9407455c3392e85cb29f4af6b954ef18d931e735890b0',
}

#: Headers a module composes ITSELF rather than through the loader's
#: EXTRA_HEADERS, pinned all the same.  urban_bem.cuh is generated by
#: tools/transcribe_urban_bem.py and compiled by woof/core/urban_bem.py
#: (04118fa24) into the BEP+BEM unit.
COMPOSED_HEADERS = {
    # LF-normalized at golden/wrf-fixes: the lane blob ended in one stray CR after
    # its final #endif; the oracle-only helper is otherwise byte-identical.
    "urban_root_uptake_480.cuh": "30d091b0ce27830798128f1bb0228c6c8101435c1bf3e530320a723386b237f8",
    "urban_ground_water_480.cuh": "94a99f3abc1a142b8197931797a5615a0cfe64a168f92d09a8ff03ac6b089dd7",
    # Direct SFIRE NVRTC compositions, with scalar native libm controls.
    "sfire_libm.cuh": "22f44144a81d60fbbd767cfcb33d260e4d1f08fdc759d962165bae70bb603ee3",
    "sfire_phys.cuh": "8424f3793b152c96d68ee9b6f112ac19e65d5f823f3deb9a6ff9c44436caa0ae",

    "urban_bem.cuh": "b4cb2ac2ed49d7e8aeb71193a191eefbe4a6247624772428dd9ac322a89cdb99",

    # FAST_SBM's direct NVRTC composition (woof/core/fast_sbm_cuda.py
    # compiles fast_sbm.cu with this directory on its include path).
    "fast_sbm_aux.cuh": "b271ebf1b4f82b24178f1c503fc05ffbf73c38d4f05bd0f1cf1c3be93826bd6f",
    "fast_sbm_coll.cuh": "c3d848392ac0d3394b5a22e8faf7865d62e1328e5f45f156fc28cdabee31d9da",
    "fast_sbm_column.cuh": "57884373b33a166c5d99c62f92bf8a4f7d380bef84bd1d7627c48abcaf19c998",
    "fast_sbm_common.cuh": "40b390032f6fcace433ccdd5d2043a4fabf63e4682b9faf31f6acf0ddb55d5a2",
    "fast_sbm_core.cuh": "c3195dd6a93fbb6189645c6c97711dcb676313da3b33c19bcb344f2817513894",
    "fast_sbm_driver.cuh": "492f0a9530bf899b74ae935dd028bffee8f32a7177067dc8b54334fa98cc083e",
    "fast_sbm_init.cuh": "5f3e2136b6afba94edd2fc36145880eb3dc6ebd902a16b3ddc938d715991fa30",
    "fast_sbm_nucl.cuh": "8cd18b15d7deb2087320e935ef767470dabd801f4e61faeeeaa3bea3c67a05ac",
    # Retained base headers composed by their dedicated runtime loaders.
    'f64cr.cuh': '3adb76fc5f559a21a457572848772debd000f77ab6557428edfa2765f2a458bb',
    'nssl2_wrf_math.cuh': '15578ff62956b321b6145d1eff2bfff88a22576ff24dc4b9d4200f9ca1b7fcb0',
    'rrtmgp_libm.cuh': '71c883fed0cffeed7ae49cd422a8519de6d1815f1c9abf1ddedd9e5039204c96',
    # Retained intake source assets. These pins assert bytes, not active use.
    'thompson_fork_constants.cuh': 'd13693dc55fbb263289b30154bcd6fbdec86b8cb67bd3c393750ee3b968f19c9',
    'gf_wrf_normalization.cuh': '2f258d559bda0b6c97a7220f2eb4c27473817ad25daf69769e858d63a1731942',
    'mynn_surface_libm.cuh': '05bff414d423ef312057ee6a3a2e6f71b9576e9fae53ec201e220b5273fac40f',
    # r290 exact-chain intake: source identity, numerical gates separate.
    'rrtmgp_planck_common_default.cuh': '4e1a8214ea8e2a3dbd88cc2cda260a21ff678d98acf4f22c971ba0b51b4eba36',
    # rad-history (29e917cbeb9): the paired LW solve derives Planck sources through this header (pk_on 2),
    # matched bitwise to the standalone kernel on 61,440,000 lay_source words.
    'rrtmgp_planck_staged_default.cuh': 'a311afa625da94395c2b7e05c99f0d6244db595136245fdee18451cfbb54ce16',
    # r290 exact-chain intake: source identity, numerical gates separate.
    'ysu_topo_default.cuh': '031efb139b827dd94fa9407455c3392e85cb29f4af6b954ef18d931e735890b0',
}
PINNED_HEADERS["sfire_libm.cuh"] = COMPOSED_HEADERS["sfire_libm.cuh"]
_merge_extension_pins(PINNED_HEADERS, _EXTENSION_PINS["headers"])
_merge_extension_pins(COMPOSED_HEADERS, _EXTENSION_PINS["composed_headers"])


@pytest.mark.parametrize("header", sorted(PINNED_HEADERS))
def test_prepended_header_is_byte_identical_to_its_pin(header):
    source = KERNELS / header
    assert source.is_file(), f"pinned CUDA header is missing: {source}"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PINNED_HEADERS[header], (
        f"prepended CUDA header {header} changed; retain its numerical authority "
        "and record the measured change before updating its pin")


def test_header_pins_cover_the_actual_loader_closure():
    from woof.core.kernels import EXTRA_HEADERS
    from woof.core.ruc_tier import RUC_FUSED_SOURCES
    used = {"common.cuh", *(header for headers in EXTRA_HEADERS.values() for header in headers),
            *RUC_FUSED_SOURCES}
    assert used == set(PINNED_HEADERS)
    # A loader may borrow an already frozen .cu unit as a header (noah_init
    # reuses Noah's FRH2O). Keep the .cuh census exact, and require every
    # borrowed .cu to carry the same source authority as its standalone unit.
    assert {path.name for path in KERNELS.glob("*.cuh")} == {
        name for name in used | set(COMPOSED_HEADERS) if name.endswith(".cuh")}
    borrowed_units = {name for name in used if name.endswith(".cu")}
    assert borrowed_units <= {f"{module}.cu" for module in PINNED}
    for name in borrowed_units:
        assert PINNED_HEADERS[name] == PINNED[name.removesuffix(".cu")]
    for header, sha in COMPOSED_HEADERS.items():
        assert hashlib.sha256((KERNELS / header).read_bytes()).hexdigest() == sha, (
            f"{header} changed; record the reading that moved it before its pin")


def test_header_fault_is_detected_even_when_module_pins_are_unchanged(tmp_path, monkeypatch):
    header = "glibc_flt32.cuh"
    (tmp_path / header).write_bytes((KERNELS / header).read_bytes() + b"\n// fault control\n")
    monkeypatch.setattr(sys.modules[__name__], "KERNELS", tmp_path)
    with pytest.raises(AssertionError, match="prepended CUDA header"):
        test_prepended_header_is_byte_identical_to_its_pin(header)


def test_the_pin_table_was_read_at_all() -> None:
    """A gate over an empty table passes and protects nothing."""

    assert len(PINNED) >= 91, (
        f"only {len(PINNED)} kernel module(s) are pinned; ninety-one were at "
        "b47a400a5 (65 frozen + 26 baseline), so a pin table was read wrongly "
        "or emptied and this gate is checking almost nothing")


def test_every_re_pinned_module_is_still_a_module_that_drifted() -> None:
    """The re-pin table cannot outlive its reason.

    If a module in ``RE_PINNED_DRIFT`` comes back into agreement with
    the historical freeze -- someone reverts the change, or re-pins
    ``FROZEN_MODULE_DIGESTS`` properly -- this entry is stale and must be
    dropped, or the file reads as if nine modules are still adrift.
    """

    reconciled = sorted(
        name for name, (sha, _c) in RE_PINNED_DRIFT.items()
        if _FROZEN.FROZEN_MODULE_DIGESTS.get(name, (None,))[0] == sha)
    assert not reconciled, (
        f"{reconciled} now agree with FROZEN_MODULE_DIGESTS; remove them "
        "from RE_PINNED_DRIFT so the drift list stays true")


@pytest.mark.parametrize("module", sorted(PINNED))
def test_the_kernel_source_is_byte_identical_to_its_pin(module: str) -> None:
    """One node per translation unit.  A tenth red is a NEW node id."""

    source = KERNELS / f"{module}.cu"
    if module == "rrtmg_lw_chain_coalesced":
        from woof.core.rrtmg_lw import (
            _LW_COALESCED_HEADER, _lw_coalesced_source)
        chain = (KERNELS / "rrtmg_lw_chain.cu").read_text(encoding="utf-8")
        payload = (_LW_COALESCED_HEADER + _lw_coalesced_source(chain)).encode()
    else:
        assert source.is_file(), (
            f"{source} is pinned and does not exist; a frozen kernel module "
            "disappeared")
        payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    where = ("RE_PINNED_DRIFT" if module in RE_PINNED_DRIFT
             else "BASELINE_PINNED in this file" if module in BASELINE_PINNED
             else "tests/test_mp8_frozen.py FROZEN_MODULE_DIGESTS")
    assert digest == PINNED[module], (
        f"woof/core/kernels/{module}.cu changed.\n"
        f"  pinned in {where}\n"
        f"    expected {PINNED[module]}\n"
        f"    actual   {digest}\n"
        "  A CUDA kernel source moved.  If the change is intended, re-pin it "
        "in the same commit and say in that commit which physics changed and "
        "what measurement shows the new behaviour is right.  A silently "
        "edited kernel runs on every column of every step and no CPU test in "
        "this repository executes it.")


def test_every_kernel_source_on_disk_is_pinned_by_something() -> None:
    """The other direction: a kernel nobody pins is a kernel nobody guards.

    ``FROZEN_MODULE_DIGESTS`` deliberately ignores kernel modules added after
    the freeze.  That is the right call for a FREEZE and the wrong one for a
    guard, because it means a new ``.cu`` -- or a renamed one -- carries no
    digest at all.  The unpinned set is listed here with its size so that
    growth is a decision.
    """

    on_disk = {path.stem for path in sorted(KERNELS.glob("*.cu"))}
    unpinned = sorted(on_disk - set(PINNED))
    assert not unpinned, (
        f"{len(unpinned)} kernel source(s) under woof/core/kernels are "
        f"pinned by no digest at all:\n  " + "\n  ".join(unpinned) +
        "\n  Every .cu in that directory was pinned at b47a400a5, so this is "
        "a CUDA translation unit that joined the product with nothing "
        "checking its bytes.  Add it to BASELINE_PINNED (or to "
        "the freeze) in the commit that adds the kernel.")


def test_mosaic_ucm_composed_source_is_pinned():
    # The composed Noah mosaic + UCM unit (NOAH_MOSAIC_UCM, urban_ucm.cu +
    # noah_mosaic.cu), bitwise against WRF v4.7.1 on the ucm and ucm_lcz
    # families (tests/test_noah_mosaic_ucm_wrf471_parity.py).  Moved when
    # urban_ucm.cu's Noah-MP-only ucm_overrides T2 arm changed; the mosaic
    # unit never launches ucm_overrides.  Previously d70a75c1.
    # WRF 4.8 water balance and the inherited source-scoped no-FTZ loader.
    # Legacy full-field hashes match the prior kernel; the new store terms
    # match the release Fortran oracle. Other WRF 4.8 physics is outside it.
    from woof.core.noah_mosaic import mosaic_ucm_source
    assert hashlib.sha256(mosaic_ucm_source().encode()).hexdigest() == (
        'cfa235dcf278a44fc9ef80dcc5700e2e6b2ddfe640c12314f0db86e4e7a65401')
