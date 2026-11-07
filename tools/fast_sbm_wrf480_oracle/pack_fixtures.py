"""Pack the fast SBM oracle streams into tests/data/fast_sbm_oracle/.

Input: the ``out/`` directory ``build.sh`` writes on the build node (copied
back whole; the runners' stdout files are not needed). Output, in ``--dest``:

* ``<family>.bin.gz``: one record stream per family, the bytes the Fortran
  harness wrote, byte for byte, compressed at gzip level 9 with no file name
  and a zero timestamp (the ``gzip -9 -n`` form, so the bytes reproduce);
* ``MANIFEST.json``: per file the build, dtypes, record layout, record
  counts, uncompressed byte count and SHA-256; the compiler, flags, source
  pins, transform diffs, control results, the KC-1 case definition and its
  acceptance counts;
* ``MANIFEST.md``: the same in plain prose.

Nothing is packed unless every check below passes: the harness files beside
this script hash as the ones that ran (``out/harness.sha256``), every stream
parses against its layout with the record counts the layout states, no float
field holds NaN or Inf, every inertness and determinism control is true, the
self-replay controls hold, both builds' KC-1 and KC-2 runs meet their case
acceptance (and the graupel build's KC-2 its own), the low-edge KC-2 run took
the pre-pass's I = 1 and J = 1 branches, every refusal case stopped where its
case says it stops, the graupel transforms are the one hail_opt
line, every milestone 1 stream is byte-identical to the one the existing
manifest in ``--dest`` records, and the packed total is at most 20 MB.

Usage::

    python pack_fixtures.py <out-dir> [--dest tests/data/fast_sbm_oracle]
                            [--reproduction <dir> --reproduction-label <text>]
                            [--allow-milestone1-change]

``--reproduction`` names a directory holding ``raw.sha256`` (``sha256sum`` of
every ``.raw`` stream under another build's ``out/``, paths relative to it),
that build's ``build_info.json`` and ``controls.json``; the manifest then
records, stream by stream, whether that compiler reproduced the bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixtures as fx  # noqa: E402

DEFAULT_DEST = fx.FIXTURE_ROOT
SIZE_LIMIT = 20 * 1024 * 1024
MIN_TEMPERATURE_K = 275.15
SWEEPS = ("polysvp", "jerrate", "jersupsat", "jerdfun", "jernewf", "onecond1",
          "nucleation", "coll_xxx", "coal_bott", "breakup", "spont", "falflux")
BUILDS = ("o64", "o32")
HARNESS_FILES = ("build.sh", "sources.sha256", "stubs_wrf.F90", "transform_access.py",
                 "instrument_probes.py", "fsbm_probe.F90", "run_dump_tables.F90",
                 "run_sweeps.F90", "run_column.F90", "transform_graupel.py", "run_kc2.F90")
COLUMN_NAMES = ("c1", "c2", "c3", "c4")

# =========================================================== milestone 2
COLD_SWEEPS = ("freez", "melt", "ice_nucl", "jernucl01_ice", "jerrate_ice", "jertimesc_ice",
               "jerdfun_ice", "onecond2", "onecond3", "coll_cold", "modkrn", "coal_cold")
M1_HARNESS_FILES = ("build.sh", "sources.sha256", "stubs_wrf.F90", "transform_access.py",
                    "instrument_probes.py", "fsbm_probe.F90", "run_dump_tables.F90",
                    "run_sweeps.F90", "run_column.F90")
KC2_COLUMNS = ("A (2,2)", "env (3,2)", "D (4,2)", "env (2,3)", "B (3,3)", "env (4,3)",
               "env (2,4)", "env (3,4)", "C (4,4)")
KC2_COLD_COLUMNS = (0, 2, 4, 8)          # tile column numbers (i fastest) of A, D, B, C
HAIL_LIVE_CALLS = (1, 3, 4, 10, 11, 12)
GRAUPEL_LIVE_CALLS = (2, 3, 5, 6, 7, 12)
CC_NAMES = ("cc01_xyz_drop_snow_hail", "cc02_xyz_drop_snow_grp", "cc03_xyx_snow_drop",
            "cc04_xyz_snow_drop_hail", "cc05_xyz_snow_drop_grp", "cc06_xyy_drop_grp",
            "cc07_xyx_grp_drop", "cc08_xyz_drop_grp_hail", "cc09_xyz_grp_drop_hail",
            "cc10_xyy_drop_hail", "cc11_xyx_hail_drop", "cc12_xxx_snow_snow")

COLD_SWEEP_TEXT = {
    "freez": "FREEZ on the seven species (liquid, three crystal habits, snow, graupel, hail): "
             "the old port's grid of 5 temperatures x 4 time steps on its draws, the split at "
             "KRFREEZ = 21 between plates and hail, no drops, and cloud and rain at three "
             "temperatures",
    "melt": "J_W_MELT on the same seven species: the old grid, every species in every rate "
            "band, the old dt = 200 s case that drives a snow bin negative (D12), the band "
            "edges at 100, 176 and 180 s, and physical snow and hail around 0 C",
    "ice_nucl": "ICE_NUCL reached through JERNUCL01_KS with the water supersaturation below "
                "zero (ICE_NUCL is private to module_mp_SBM_Nucleation; the probe build "
                "records its own call): the old port's 9 habit and Meyers boundaries x 2 "
                "SUP2_OLD, either side of each boundary, the -37 C limit, both "
                "DELSUPICE_MAX clamps, a falling supersaturation, and crystals above HELEK1",
    "jernucl01_ice": "JERNUCL01_KS with ice (ICEPROCS 1): the old port's 48 cases, 8 "
                     "temperatures x 3 supersaturation pairs x cloud base or not, with "
                     "homogeneous freezing and ICE_NUCL read back through the probe build",
    "jerrate_ice": "ice JERRATE_KS: three crystal habits, snow (whose nskin rule switches "
                   "at 273.15 K), graupel and hail with liquid fraction 0 and 0.5, snow with "
                   "IN 6, at 4 temperatures and 3 pressures",
    "jertimesc_ice": "ice JERTIMESC_KS: the crystal habits (ID 3) and snow, graupel and hail "
                     "(ID 1), on the old port's draws and on physical spectra",
    "jerdfun_ice": "ice JERDFUN_KS (the linear mass law of ITYPE 2 to 5 and the crystal "
                   "habits) at 5 values of DEL2N, the old port's (IND 2, ITYPE 3) cases, and "
                   "a negative bin that the remap drops without refusing",
    "onecond2": "ONECOND2 in KCOND 20 and 21 (the exact zero included) at 4 temperatures, 5 "
                "ice supersaturations and 3 ice populations, and two dilutions that reach "
                "JERSUPSAT_KS's slow-ice and no-rate branches; the JERSUPSAT_KS call inside "
                "is read back through the probe build",
    "onecond3": "ONECOND3 in KCOND 30, 31, 32 and 50 (DEL1N > 0 > DEL2N and both exact "
                "zeros) at 3 temperatures and 3 populations, a dilute case (JERSUPSAT_KS "
                "branch 7, finding S2), a trace of ice (branch 2), and the skip above "
                "-0.187 C",
    "coll_cold": "coll_xyx_lwf, coll_xyy_lwf and coll_xyz_lwf: the old port's 24 live cases "
                 "(record 'orient') on the synthetic grid x = 1e-9 2^(k-1) with y = 1.25 x "
                 "(set 1, the orientation cases), on the synthetic y = x (set 2) and on the "
                 "real tables, x = y = xl_mg with the module's chucm and ima (set 3), with "
                 "the same draws in every set; and the six calls COAL_BOTT_NEW makes "
                 "(record 'table'), on the real grids and kernels at 850 and 500 hPa",
    "modkrn": "modkrn_KS CASE 11 (and INTERPOL_SE through it): the old port's 12 cases and "
              "the seven table nodes; the output starts at -1, so each record shows whether "
              "it was set",
    "coal_cold": "COAL_BOTT_NEW with ice in the hail build: 5 populations at 11 temperatures "
                 "from 273.15 to 233.15 K through the Hallett-Mossop ramp edges, at 600 hPa, "
                 "and the mixed population at 850 and 400 hPa; the cold block's collision "
                 "calls, Hallett-Mossop values and riming heat are read back through the "
                 "probe build. The field colleff is WRF's CollEff = PRDKRN (:8322) only "
                 "where colleff_defined is 1: modkrn_KS leaves PRDKRN unset above 0 C in "
                 "the build's arithmetic (D6; 273.15 K is above it in the native build, "
                 "where tc = 273.15d0 - real(273.15, 4) = +6.1e-6) and at exactly -40 C, so "
                 "COAL_BOTT_NEW then returns an uninitialised value, and the record holds 0 "
                 "with colleff_defined 0; gate colleff only where it is 1",
    "coal_cold_graupel": "the same COAL_BOTT_NEW cases in the graupel build (dense ice "
                         "passed as FF4R), with the same colleff_defined caveat",
}

# Cold refusal cases: routine, input, expected stop per build, pristine line of
# the stop, and whether FAST_SBM can reach the input.
REFUSAL_COLD = {
    1: ("JERNUCL01_KS/ICE_NUCL",
        "-15 C with a habit-3 grid whose first mass is the smallest subnormal of the "
        "build's real: 1/DX overflows and the concentration check sees an infinite change",
        {"o64": "bare STOP 099", "o32": "returned"}, 3463, False),
    2: ("JERDFUN_KS/JERNEWF_KS", "a negative snow bin and no growth",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 2750, False),
    3: ("ONECOND2", "no vapour (DEL1N = DEL2N = -1) with snow: RW and PW are NaN",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 7441, False),
    4: ("ONECOND2", "a negative substep at 10 percent ice supersaturation (KCOND 21)",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 7452, False),
    5: ("ONECOND2", "200 percent ice supersaturation over dense snow for 3600 s: a "
        "deposition heating above 5 K",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 7694, False),
    6: ("ONECOND3", "no vapour with drops and snow: RW and PW are NaN",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 7993, False),
    7: ("ONECOND3", "a negative substep",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 8049, False),
    8: ("ONECOND3", "200 percent water supersaturation over a dense cloud and snow for "
        "3600 s: a heating above 5 K",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 8239, False),
    9: ("COAL_BOTT_NEW", "a flat, very dilute drop spectrum over a 1e4 s step at 263 K: the "
        "gmin floor of coll_xxx_lwf adds more than 5 percent to the drops",
        {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 8599, False),
    10: ("COAL_BOTT_NEW", "a dense cloud with drizzle rimed by much hail over 600 s at "
         "400 hPa: a riming heat above 5 K",
         {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 8636, False),
    11: ("COAL_BOTT_NEW", "a negative snow bin at 288 K: the snow check at the end of the step",
         {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 8614, False),
    12: ("ONECOND2", "a negative substep at 10 percent below ice saturation (KCOND 20)",
         {"o64": "wrf_error_fatal", "o32": "wrf_error_fatal"}, 7535, False),
    13: ("JERNUCL01_KS/ICE_NUCL",
         "-15 C at 60 percent ice supersaturation with a finite mixed-sign habit-3 spectrum "
         "whose concentration terms are about 7.5, +1e17 and -1.00001e17 per cm3: rounding "
         "in the real(8) sum turns the 1.106 per cm3 of nucleation into a change of 16",
         {"o64": "bare STOP 099", "o32": "bare STOP 099"}, 3463, False),
}

# What a bare STOP 099 must show besides its exit status, so it cannot be a
# stub wrf_error_fatal whose refusal-file write failed: ICE_NUCL's last print
# before the stop (:3462) on stdout, and no wrf_error_fatal line on stderr.
STOP_099_PRINT = "STOP 099 : DELCONCI_AFNUCL(ICE) > 10/cm^3"

FINDINGS_COLD = {
    "ice_nucl_stop_099_unreachable": (
        "ICE_NUCL stops with a bare STOP 099 (:3463) when a habit's crystal concentration, "
        "the running real(8) sum of 3 COL PSI2 xi over all 33 bins (:3375-3381, "
        ":3446-3452), changes by more than 10 per cm3 in one call. Nucleation adds DELTAF, "
        "at most HELEK1 = C1 exp(A1 + B1 DELSUPICE_MAX) = 1.106 per cm3, to bin 1 of one "
        "habit, and only when that habit's concentration is at most HELEK1 (:3435). With "
        "nonnegative bins that bounds every term, so the change stays at most 1.106. Finite "
        "bins of mixed sign reach the stop through rounding in the sum: a term of +1e17 "
        "(where the sum's spacing is 16) and a slightly larger negative term keep the "
        "concentration negative, and a bin-1 term just under 8 is absorbed before "
        "nucleation and adds 16 after it (refusal case 13 stops in both builds). A "
        "non-finite change reaches it too: a subnormal first grid mass makes 1/DX overflow "
        "(refusal case 1, the promoted build only; from a float32 grid the product cannot "
        "underflow in real(8), so the native build returns). FAST_SBM cannot reach the stop "
        "because it passes JERNUCL01_KS no crystals: FF2R is zeroed at every point "
        "(:4548-4550) and again after every nucleation call (:4667-4674), so FF2IN is zero "
        "on every call. The only crystals ICE_NUCL can then see are drops that "
        "JERNUCL01_KS's homogeneous freezing (:3101-3110) put into habit 2 in the same "
        "call, which happens only when its latent heat lifts the point from -38 C or below "
        "to -37 C or above; the nucleating habit is then habit 1 (below -22.4 C), which is "
        "zero, and habit 2 does not change."),
    "modkrn_unset_at_minus_40": (
        "modkrn_KS CASE 11 (:1481-1498) sets its output through INTERPOL_SE (:1362-1393), "
        "which assigns nothing when H equals its last node (-40 C): every test is strict "
        "and the last node is the smallest. At exactly -40 C the output keeps whatever the "
        "caller held (-1 in the sweep), as above 0 C, where modkrn_KS returns first "
        "(:1416-1417, finding D6). FAST_SBM calls COAL_BOTT_NEW for TT >= 233.15 K, so a "
        "point at exactly -40 C collides snow with snow using an undefined PRDKRN, and "
        "COAL_BOTT_NEW returns CollEff = PRDKRN (:8322) undefined there too."),
    "jerdfun_negative_bin_with_growth": (
        "JERNEWF_KS's Kovetz-Olund remap (:2537-2573) moves only bins with FI > 0, so when "
        "any bin grows a negative bin is dropped without a refusal (the jerdfun_ice sweep's "
        "set 3). Its negative check (:2735-2751) fires when nothing grows (refusal case 2), "
        "and also when a bin's new mass passes the top of the remap's grid, "
        "RRS(NRX+1) = 1024 RRS(NRX) (:2515): the bracketing search then stops at the top "
        "bin, GMAT = (RRP - RN)/(RRP - RRTMP) is negative (:2564-2565) and the top bin can "
        "go negative. That needs a bin to grow by more than a factor of 1024 in mass in one "
        "call, which is extreme but not excluded. JERDFUN_KS's own check (:2426-2429) tests "
        "the same values after that check and so can never fire."),
    "deldrop_reached_only_through_the_gmin_floor": (
        "In COAL_BOTT_NEW the drops lose mass in every cold collision, and the breakup step "
        "keeps a gain of at most 0.1 percent; the only other source of drop mass is "
        "coll_xxx_lwf's floor, which raises a bin left below gmin = 1e-13 to gmin. The "
        "deldrop check (:8598) fires only when those floors add more than 5 percent, which "
        "needs a very dilute spectrum and a very long step (refusal case 9)."),
    "snow_breakup_guard_always_true": (
        "The snow-breakup guard (:4836) sums the slice FF3R(KR_SNOW_MIN:NKR), empty on the "
        "33-bin grid (KR_SNOW_MIN = 34), so it is 0 > (33 - 34) 1e-30 and true at every "
        "point; BreakUp_Snow is then skipped because KR_SNOW_MAX = 35 > NKR. KC-2's "
        "counts show the guard true at all 15000 points of every tile column and no call."),
    "onecond2_sign_flip_checks_unreachable": (
        "ONECOND2's two sign checks (:7522-7525 and :7608-7611) cannot fire: its drop rates "
        "are zero, so JERSUPSAT_KS takes the ice-only branch (or the no-rate branch), where "
        "DEL2N = DEL2 exp(-PI DT) keeps DEL2's sign, the dynamic terms being zero on every "
        "WRF call."),
    "supsat_small_ratio_branches_unreachable_from_onecond3": (
        "JERSUPSAT_KS's branches 8 and 9 (finding S1) need RI/RW at least 1e12 (or at most "
        "1e-12) while the only-ice and only-water tests pass. In ONECOND3 RI/RW and PI/PW "
        "are the same ratio of ice to drop relaxation rates up to factors between 1 and 1.2, "
        "so for nonnegative spectra a ratio of 1e12 is sent to the only-ice (or only-water) "
        "branch first. Over ONECOND2 and ONECOND3, KC-2 reaches branches 0, 2 and 4 to 7 in "
        "both builds and precisions and never 8 or 9 (kc2.catalogue_counts); S2 (branch 7) "
        "is reached by real ONECOND3 calls."),
    "graupel_build_dead_calls": (
        "In the graupel build FF5R is zero whenever COAL_BOTT_NEW runs (FREEZ's hail is "
        "merged into graupel after the collisions), so the drop-hail calls (:8524, :8534) "
        "never run; the two drop-graupel-to-hail calls (:8501, :8512) need ALWC of at least "
        "ALCR_G = 100 g/m3. In the hail build the graupel calls never run. KC-2 and KC-2g "
        "count each call site."),
    "anti_bergeron_regime": (
        "ONECOND3's KCOND 50 (:8006-8016) prints 'Anti-Bregeron Regime, No DIFFU' and "
        "returns without growth. It needs DEL1N > 0 > DEL2N or an exact zero. FAST_SBM's "
        "saturation fits (2.53e12 exp(-5420/T) over water, 3.41e13 exp(-6130/T) over ice) "
        "cross at 272.96 K, about -0.186 C, and below that the water saturation pressure "
        "is the larger, so DEL1N > 0 > DEL2N is excluded wherever ONECOND3 is dispatched "
        "(below -0.187 C, :7908); between the two it is not. The sweep reaches KCOND 50 "
        "with stated inputs, KC-2 never does."),
}


def refusal_container(out: Path, name: str, sub: str, cases: dict, access_source: str,
                      build_expect=None, family: str = "refusals_cold") -> tuple[bytes, dict]:
    """Pack a refusals container; ``build_expect(n, b)`` is the expected stop kind."""

    members = []
    container = bytearray()
    for b in BUILDS:
        for n in sorted(cases):
            base = out / b / sub
            stem = f"{name}_{n}"
            raw, layout, records = read_stream(base, stem)
            status = int((base / f"{stem}.status").read_text().strip())
            err = (base / f"{stem}.err").read_text(errors="replace")
            msg_path = base / f"{stem}.message"
            message = msg_path.read_text().strip() if msg_path.exists() else ""
            if f"FSBM_REFUSAL_CALLING {n}" not in err:
                raise PackRefusal(f"{name} {n} ({b}) never reached its call")
            if status == 99 and message:
                kind = "wrf_error_fatal"
                if f"WRF_ERROR_FATAL: {message}" not in err:
                    raise PackRefusal(f"{name} {n} ({b}): the message is not on stderr")
            elif status == 99:
                # A stub wrf_error_fatal whose refusal-file write failed would
                # also exit 99 without a message file: ICE_NUCL's own stop has
                # no wrf_error_fatal line and printed its last line first.
                kind = "bare STOP 099"
                out_path = base / f"{stem}.out"
                if not out_path.exists():
                    raise PackRefusal(f"{name} {n} ({b}): exit 99 without a message and no "
                                      "stdout file to tell ICE_NUCL's stop apart")
                if "WRF_ERROR_FATAL:" in err:
                    raise PackRefusal(f"{name} {n} ({b}): exit 99 with a wrf_error_fatal "
                                      "line but no message file")
                if STOP_099_PRINT not in out_path.read_text(errors="replace"):
                    raise PackRefusal(f"{name} {n} ({b}): exit 99 without ICE_NUCL's "
                                      "'STOP 099' print")
            elif status == 0 and not message:
                kind = "bare STOP"
            elif status == 3 and "FSBM_REFUSAL_RETURNED" in err and not message:
                kind = "returned"
            else:
                raise PackRefusal(f"{name} {n} ({b}): exit status {status}")
            expected = build_expect(n, b)
            if kind != expected:
                raise PackRefusal(f"{name} {n} ({b}): {kind}, expected {expected}")
            routine, what, _, line, reachable = cases[n]
            wrf_line = wrf_line_of(message, access_source) if message else line
            if kind in ("wrf_error_fatal", "bare STOP 099") and wrf_line != line:
                raise PackRefusal(f"{name} {n} ({b}): stopped at :{wrf_line}, expected :{line}")
            members.append({
                "case": n, "build": b, "routine": routine, "input": what,
                "exit_status": status, "message": message, "stop_kind": kind,
                "wrf_line": wrf_line if kind != "returned" else None,
                "reachable_from_fast_sbm": reachable,
                "offset": len(container), "bytes": len(raw), "sha256": sha256(raw),
                "layout": layout.to_json(),
            })
            container += raw
    container = bytes(container)
    entry = {
        "file": f"{family}.bin.gz", "family": family, "build": "o64+o32",
        "container": "refusals",
        "format": "concatenated record streams, one per case and build; each member's "
                  "offset, length, SHA-256 and layout are listed",
        "dtype": "record streams",
        "bytes": len(container), "sha256": sha256(container),
        "record_count": sum(sum(t["count"] for t in m["layout"]) for m in members),
        "members": members,
    }
    return container, entry


def kc2_acceptance(recs: list, graupel: bool) -> dict:
    acc = records_of(recs, "acceptance")[0]
    counts = {k: [int(x) for x in np.asarray(acc[k]).ravel(order="F")] for k in acc.fields
              if np.asarray(acc[k]).dtype.kind == "i" and k != "column_type"}
    tmin = [float(x) for x in np.asarray(acc["tmin_run"]).ravel(order="F")]
    tmax = [float(x) for x in np.asarray(acc["tmax_run"]).ravel(order="F")]
    snow = [float(x) for x in np.asarray(acc["snownc_final"]).ravel(order="F")]
    dense = [float(x) for x in np.asarray(acc["graupelnc_final"]).ravel(order="F")]
    rain = [float(x) for x in np.asarray(acc["rainnc_final"]).ravel(order="F")]
    live = GRAUPEL_LIVE_CALLS if graupel else HAIL_LIVE_CALLS
    dead = [i for i in range(1, 13) if i not in live]

    def total(key):
        return sum(counts[key])

    checks = {
        "through_0_C_and_below_minus_40_C": max(tmax) >= 295.0 and min(tmin) < 233.15,
        "every_habit_band_calls_ice_nucl": all(total(f"icenucl_band{b}") > 0 for b in range(1, 7)),
        "ice_nucleated_in_bands_2_to_6": all(total(f"icenucl_add_band{b}") > 0 for b in range(2, 7)),
        "homogeneous_freezing": total("homfrz") > 0,
        "onecond1": total("cond1_calls") > 0,
        "onecond2": total("cond2_calls") > 0,
        "onecond3": total("cond3_calls") > 0,
        "onecond2_kcond20_and_21": total("cond2_kcond20") > 0 and total("cond2_kcond21") > 0,
        "kcond_30_31_32": all(total(f"cond3_kcond{k}") > 0 for k in (30, 31, 32)),
        "every_live_cold_collision_call": all(total(CC_NAMES[i - 1]) > 0 for i in live),
        "no_dead_cold_collision_call": all(total(CC_NAMES[i - 1]) == 0 for i in dead),
        "hallett_mossop_splinters": total("hm_splinters") > 0,
        "riming_heat": total("rime_heat") > 0,
        "freez_to_plates": total("freez_to_plates") > 0,
        "freez_to_hail": total("freez_to_hail") > 0,
        "every_melt_band": all(total(f"melt_snow_band{b}") > 0 and total(f"melt_dense_band{b}") > 0
                               for b in (1, 2, 3)),
        "d11_fires": total("d11_levels") > 0,
        "d15_points": total("d15_points") > 0,
        "breakup_snow_never_called": total("snowbrk_call") == 0,
        "snow_breakup_guard_true_everywhere": all(c == 300 * 50 for c in counts["snowbrk_guard_true"]),
        "surface_snow": max(snow) > 0,
        "surface_dense_ice": max(dense) > 0,
        "surface_rain": max(rain) > 0,
    }
    catalogue = {
        "D11_levels_small_snow_deleted": total("d11_levels"),
        "D12_negative_bins_after_melt": total("melt_negative_bin"),
        "D14_homogeneous_freezing_events": total("homfrz"),
        "D15_prepass_points_below_238K": total("d15_points"),
        "D16_mixed_phase_at_or_below_233K_no_growth": total("dispatch_none_d16_cold"),
        "D16_ice_without_drops_above_minus_0.187C_no_growth": total("dispatch_none_d16_warm"),
        "S1_supsat_branches_8_and_9": total("supsat_branch8") + total("supsat_branch9"),
        "S2_supsat_branch_7": total("supsat_branch7"),
        "supsat_branches": {str(i): total(f"supsat_branch{i}") for i in range(10)},
        "kcond50_calls": total("cond3_kcond50"),
    }
    return {"columns": list(KC2_COLUMNS), "min_temperature_k": tmin, "max_temperature_k": tmax,
            "rainnc_final_mm": rain, "snownc_final_mm": snow, "graupelnc_final_mm": dense,
            "counts": counts, "checks": checks, "catalogue_counts": catalogue}


def kc2edge_acceptance(replay: list, probe: list) -> dict:
    """The low-edge KC-2 run took the pre-pass's I = 1 and J = 1 branches."""

    case = records_of(replay, "edge_case")[0]
    prepass = records_of(probe, "prepass")
    counts = records_of(probe, "counts_tile")[0]
    pairs = [(int(r["step"]), r) for r in records_of(replay, "pre")]
    posts = {int(r["step"]): r for r in records_of(replay, "post")}
    ds = [np.asarray(r["dsupice_xyz"]) for r in prepass]       # (i, k, j) from its, jts
    dtf = [np.asarray(r["dtfreez_xyz"]) for r in prepass]
    types = np.asarray(case["column_type"])
    icenucl = [int(x) for x in np.asarray(counts["icenucl_calls"])]
    checks = {
        "tile_is_i_j_1_to_3": [int(x) for x in case["tile"]] == [1, 3, 1, 3, 1, 50],
        "corner_column_is_hallett_mossop": int(types[0, 0]) == 2,
        "prepass_recorded_at_the_replay_steps": [int(r["step"]) for r in prepass]
        == [int(x) for x in case["replay_steps"]] == [s for s, _ in pairs],
        "dsupice_i1_branch_taken": any(np.any(d[0, :, :] != 0) for d in ds),
        "dsupice_j1_branch_taken": any(np.any(d[:, :, 0] != 0) for d in ds),
        "dtfreez_i1_branch_taken": any(np.any(d[0, :, :] != 0) for d in dtf),
        "dtfreez_j1_branch_taken": any(np.any(d[:, :, 0] != 0) for d in dtf),
        "ice_nucl_called_in_the_corner": icenucl[0] > 0,
        "corner_state_changes": any(
            not np.array_equal(np.asarray(pre["th"])[0, :, 0], np.asarray(posts[s]["th"])[0, :, 0])
            for s, pre in pairs),
    }
    return {
        "tile": [int(x) for x in case["tile"]],
        "replay_steps": [int(x) for x in case["replay_steps"]],
        "steps": int(case["nsteps"]),
        "dsupice_nonzero_points_i1": int(sum(np.count_nonzero(d[0, :, :]) for d in ds)),
        "dsupice_nonzero_points_j1": int(sum(np.count_nonzero(d[:, :, 0]) for d in ds)),
        "dtfreez_nonzero_points_i1": int(sum(np.count_nonzero(d[0, :, :]) for d in dtf)),
        "dtfreez_nonzero_points_j1": int(sum(np.count_nonzero(d[:, :, 0]) for d in dtf)),
        "icenucl_calls": icenucl,
        "checks": checks,
    }


KC2EDGE_DEFINITION = {
    "tile": "the KC-2 domain (ids..ide and jds..jde 1..5, memory 0..6) with the tile moved "
            "to i, j = 1..3, called together in one FAST_SBM call, so the pre-pass takes "
            "its I = 1 and J = 1 branches: DERIVT_X and DERIVT_Y (:4422-4423, :4429-4430) "
            "and the DSUPICE_XYZ differences (:4460-4462, :4468-4469, :4483-4485, "
            ":4491-4492); the halo is the column i = 4 and the row j = 4",
    "columns": "KC-2's columns (A at (2,2) and B at (3,3) in the tile; D at (4,2) and C at "
               "(4,4) in the halo) plus a second Hallett-Mossop column, type B, in the "
               "corner (1,1): the DSUPICE_XYZ branch (:4441) needs 2 percent ice "
               "supersaturation below -5 C at the point itself, which KC-2's environment "
               "columns never reach; every other column as in KC-2",
    "run": "6 calls of 12 s from the KC-2 initial state, advected between calls as in KC-2; "
           "replay records (the static fields, then pre and post) at steps 2, 4 and 6, and "
           "the pre-pass arrays of the tile at the same steps from the probe build (no "
           "per-column probe records)",
    "purpose": "an oracle for the host pre-pass at a real WRF tile's low edge (its = ids, "
               "jts = jds); KC-2 itself starts at i, j = 2",
}


def kc2_definition(case: fx.Record) -> dict:
    return {
        "tile": "a 3 x 3 tile, i and j 2..4, in the corner of a 4 x 4 domain (ids..ide and "
                "jds..jde 1..5), memory 0..6, called together in one FAST_SBM call; the "
                "pre-pass reads the halo row j = 1 and column i = 1 and takes its IDE-1 and "
                "JDE-1 edge branches at i or j = 4",
        "columns": {
            "A (2,2)": "deep mixed-phase updraft: surface 300 K, 7 K/km, w1 = 4 m/s "
                       "sin(pi z / H) for the whole hour; RH 92 percent below 1.5 km, 85 "
                       "percent to 0 C, ice-saturated to 11 km, 70 percent of ice saturation "
                       "above",
            "B (3,3)": "Hallett-Mossop updraft: 298 K, 6.5 K/km, w1 = 1.5 m/s for 1800 s; a "
                       "water-saturated cloud layer 3.5-6.5 km holding a supercooled cloud "
                       "(250 cm-3, 9 um) with a 2 mm/h Marshall-Palmer tail from 3.8 to "
                       "6 km, dense ice (bins 12-26) from 4.2 to 5.5 km and snow (bins 8-28) "
                       "from 5 to 6.5 km",
            "C (4,4)": "melting stratiform column: 275.5 K (0 C near 390 m), 6 K/km, "
                       "w1 = 0.2 m/s; snow over every bin between 1.5 and 5 km (ramped over "
                       "the lowest kilometre), ice-saturated to 5.5 km; the last tile column",
            "D (4,2)": "hail shaft: 299 K, 7 K/km, w1 = -0.8 m/s (a downdraft); large dense "
                       "ice (bins 18-33) between 5 and 8 km, a supercooled cloud (150 cm-3, "
                       "7 um) between 4 and 6.5 km and light snow between 6 and 8 km",
            "environment": "the other five tile columns and the seven halo columns: still, "
                           "no hydrometeors, surface 299 K falling 0.8 K per column in x and "
                           "0.5 K in y, 6.8 K/km, RH 75/60/45 percent below 1.5 km, to 4 km "
                           "and above, falling 4 percent per column in y",
        },
        "grid": "50 levels, lowest layer 134 m, each layer 2.5 percent deeper than the one "
                "below, top 13.06 km; w on the 51 full levels, zero at the bottom and top",
        "sounding": "T = Tsfc - lapse z, p hydrostatic for that lapse from 1000 hPa with "
                    "g = 9.81, Rd = 287, cp = 1004.5, Exner (p/1e5)^(Rd/cp); qv from the "
                    "relative humidity over water with es = 2.53e12 exp(-5420/T) and "
                    "ei = 3.41e13 exp(-6130/T) dyn/cm2",
        "winds": "u = 6 + 1.5 (i-1) + 0.5 (j-1) + 0.1 k m/s, v = -2 + 0.5 (i-1) - (j-1) + "
                 "0.05 k m/s, prescribed and constant; they enter only the ice pre-pass",
        "time_step_s": float(case["dt"]),
        "steps": int(case["nsteps"]),
        "advection": "KC-1's operator (the flux form with its divergence term restored, "
                     "first-order upwind, fixed density, zero flux at the bottom and top) on "
                     "theta, qv and all 142 bins of the tile columns; th_old and qv_old are "
                     "not advected; the halo never changes",
        "replay_steps": [int(x) for x in case["replay_steps"]],
        "probe_steps": [int(x) for x in case["probe_steps"]],
        "probe_levels": [int(x) for x in case["probe_levels"]],
        "probe_columns": ["A (2,2)", "D (4,2)", "B (3,3)", "C (4,4)"],
        "profile_every": int(case["profile_every"]),
        "call": "FAST_SBM(itimestep = n, dt = 12, dx = dy = 1000, sbm_diagnostics = 0, "
                "diagflag absent, num_sbmradar = 1) after FAST_HUCMINIT(12)",
    }


BUILD_ROLES = {
    "o64": "WRF source promoted to float64 (-freal-4-real-8): the tight gate for "
           "the float64 host reference",
    "o32": "WRF as it runs (default real float32, the declared REAL(KIND=8) "
           "locals float64): compared through the native envelope",
    "o32_wrf": "evidence only, never gated: the native build at WRF's production "
               "optimisation, to record what those flags change",
}

SWEEP_TEXT = {
    "polysvp": "POLYSVP over 233-313 K in 0.25 K steps plus six edge values, water and ice",
    "jerrate": "JERRATE_KS for drops at 4 temperatures and 4 pressures",
    "jersupsat": "JERSUPSAT_KS in every branch (water only on both sides of |RW dt| = 1e-6, "
                 "ice only on both sides of |PI dt| = 1e-6, both phases on both sides of "
                 "|GAMA dt| = 1e-6 with each small-ratio sub-branch, the G2 clamp), two time "
                 "steps, with and without dynamic forcing; branch labels are the harness's "
                 "classification with the routine's own predicates",
    "jerdfun": "JERDFUN_KS (IND 1, the m^(2/3) law) for growth and evaporation, small and "
               "large, both IDROP settings",
    "jernewf": "JERNEWF_KS reached through JERDFUN_KS with IND 2: small and large growth, "
               "mixed growth, evaporation, evaporation below bin 1, no change, the 3-point "
               "path and the bins 6-12 tail remap",
    "onecond1": "ONECOND1 in condensation and evaporation, 2 temperatures, 5 supersaturations, "
                "4 spectra",
    "nucleation": "JERNUCL01_KS warm: 3 temperatures, 5 supersaturations, both activation "
                  "branches, 3 updrafts, fresh continental, depleted continental with drops, "
                  "fresh maritime",
    "coll_xxx": "coll_xxx_lwf with cwll from Kernals_KS for cloud, cloud plus rain and rain "
                "at 4 pressures and 3 collision time steps",
    "coal_bott": "COAL_BOTT_NEW, same grid of cases plus breakup stress cases; the breakup "
                 "path, final ndiv and the drop spectrum before and after coll_xxx_lwf and "
                 "after breakup are read back through the probe build",
    "breakup": "coll_breakup_KS single calls at 4 time steps and the substep sequences "
               "COAL_BOTT_NEW uses (ndiv 1, 2, 4, 8)",
    "spont": "Spont_Rain_BreakUp over 6 spectra (including the two early-return paths) "
             "and 2 time steps",
    "falflux": "FALFLUXHUCM_Z on a uniform 100 m and the KC-1 stretched grid, 2 profiles, "
               "2 time steps, several substeps each",
}

REFUSAL_TEXT = {
    1: ("ONECOND1", "condensation substep with a dense large-droplet cloud at 50 percent "
        "supersaturation: the per-substep latent heating limit"),
    2: ("ONECOND1", "the same cloud at 10 percent relative humidity (evaporation branch, "
        "KCOND 10): the same latent heating limit"),
    3: ("JERDFUN_KS/JERNEWF_KS", "a negative bin with no growth: the negative spectrum check"),
    4: ("COAL_BOTT_NEW", "a negative bin no collision pair targets: the negative drops "
        "check at the end of the collision step"),
    5: ("JERNUCL01_KS/Cloud_Base_Super", "aerosol in every bin and a vanishing updraft: "
        "the cloud-base activation search finds no bin"),
    6: ("FALFLUXHUCM_Z", "zero fall speeds: a bare STOP, which ends the process with exit "
        "status 0 and no message rather than through wrf_error_fatal"),
}


class PackRefusal(RuntimeError):
    """The build output is not fit to become the oracle fixtures."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    # Level 9, no name, zero mtime: the gzip -9 -n form, reproducible.
    return gzip.compress(data, compresslevel=9, mtime=0)


def read_stream(base: Path, rel: str) -> tuple[bytes, fx.Layout, list[fx.Record]]:
    raw = (base / f"{rel}.raw").read_bytes()
    layout = fx.parse_layout((base / f"{rel}.layout").read_text())
    if layout.nbytes != len(raw):
        raise PackRefusal(f"{rel}: layout accounts for {layout.nbytes} bytes, stream has {len(raw)}")
    records = list(fx.parse_stream(raw, layout))
    for rec in records:
        for key, value in rec.fields.items():
            arr = np.asarray(value)
            if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
                raise PackRefusal(f"{rel}: {rec.type}.{key} holds NaN or Inf")
    return raw, layout, records


def check_build_kinds(name: str, build: str, layout: fx.Layout) -> dict:
    kinds = sorted({f.kind for t in layout.types for f in t.fields})
    if build == "o64" and "f4" in kinds:
        raise PackRefusal(f"{name}: the promoted build wrote a float32 field")
    return {"field_kinds": kinds,
            "dtypes": sorted({fx.KIND_DTYPES[k] for k in kinds} | {"<i4"})}


def check_harness(out: Path) -> dict:
    recorded = {}
    for line in (out / "harness.sha256").read_text().splitlines():
        digest, name = line.split()
        recorded[name.lstrip("*")] = digest
    for name in HARNESS_FILES:
        mine = sha256((HERE / name).read_bytes())
        if recorded.get(name) != mine:
            raise PackRefusal(f"{name} beside pack_fixtures.py is not the file that ran "
                              f"({mine} against {recorded.get(name)})")
    return {name: recorded[name] for name in HARNESS_FILES}


def parse_pins(text: str) -> dict:
    pins = {"sources": {}, "archive": None, "members": {}, "url": None}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "url":
            pins["url"] = parts[1]
        elif parts[0] == "source":
            pins["sources"][parts[3]] = {"lf_sha256": parts[1]}
        elif parts[0] == "archive":
            pins["archive"] = {"file": parts[3], "sha256": parts[1], "bytes": int(parts[2])}
        elif parts[0] == "member":
            pins["members"][parts[3]] = {"sha256": parts[1], "bytes": int(parts[2])}
    return pins


def diff_body(diff: str) -> tuple[list[str], list[str], list[str]]:
    lines = diff.splitlines()
    hunks = [ln for ln in lines if ln.startswith("@@")]
    body = [ln for ln in lines if not ln.startswith(("---", "+++", "@@"))]
    return hunks, [b[1:] for b in body if b.startswith("-")], [b[1:] for b in body if b.startswith("+")]


def wrf_line_of(message: str, access_source: str) -> int | None:
    """Pristine line number of the check that writes ``message``."""

    if not message:
        return None
    hits = [i for i, line in enumerate(access_source.split("\n"), start=1)
            if message in line and "wrf_error_fatal" in line.lower()]
    if len(hits) != 1:
        raise PackRefusal(f"refusal message {message!r} found on {len(hits)} source lines")
    # The access transform removed pristine line 3862; later lines moved up one.
    return hits[0] + 1 if hits[0] >= 3862 else hits[0]


def records_of(records: list[fx.Record], rtype: str) -> list[fx.Record]:
    return [r for r in records if r.type == rtype]


def fields_equal(a: fx.Record, b: fx.Record) -> bool:
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a.fields)


# --------------------------------------------------------------------- packing
def reproduction_record(out: Path, repro: Path, label: str) -> dict:
    def load(path: Path) -> dict[str, str]:
        table = {}
        for line in path.read_text().splitlines():
            if line.strip():
                digest, name = line.split()
                table[name.lstrip("*").removeprefix("./")] = digest
        return table

    theirs = load(repro / "raw.sha256")
    mine = {p.relative_to(out).as_posix(): sha256(p.read_bytes())
            for p in sorted(out.rglob("*.raw"))}
    info = json.loads((repro / "build_info.json").read_text())
    controls = json.loads((repro / "controls.json").read_text())
    same = sorted(k for k in mine if theirs.get(k) == mine[k])
    differ = sorted(k for k in mine if k in theirs and theirs[k] != mine[k])
    return {
        "label": label,
        "compiler": info["compiler"],
        "kernel": info["kernel"],
        "cpu": info["cpu"],
        "date_utc": info["date_utc"],
        "flags_identical": info["flags"] == json.loads((out / "build_info.json").read_text())["flags"],
        "streams_compared": len(mine),
        "streams_identical": len(same),
        "streams_differing": differ,
        "streams_missing": sorted(set(mine) ^ set(theirs)),
        "controls_all_true": all(v is True for k, v in controls.items()
                                 if k.startswith(("determinism_", "inert_"))),
        "gated": False,
    }


def pack(out: Path, dest: Path, repro: Path | None = None, repro_label: str = "",
         allow_m1_change: bool = False) -> dict:
    out = out.resolve()
    previous = None
    if (dest / "MANIFEST.json").exists():
        previous = json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))
    harness = check_harness(out)
    build_info = json.loads((out / "build_info.json").read_text())
    controls = json.loads((out / "controls.json").read_text())
    pins = parse_pins((HERE / "sources.sha256").read_text())
    access_source = (out / "module_mp_fast_sbm_access.F").read_text()

    files: dict[str, dict] = {}
    blobs: dict[str, bytes] = {}
    parsed: dict[str, list[fx.Record]] = {}

    def add(name: str, rel: str, build: str, family: str, description: str, **extra):
        raw, layout, records = read_stream(out, rel)
        info = check_build_kinds(name, build, layout)
        files[name] = {
            "file": f"{name}.bin.gz",
            "family": family,
            "build": build,
            "format": "fsbm record stream: int32 type tag, then the fields, raw "
                      "little-endian, arrays in Fortran order",
            "dtype": "record stream",
            **info,
            "bytes": len(raw),
            "sha256": sha256(raw),
            "record_count": sum(t.count for t in layout.types),
            "record_counts": {t.name: t.count for t in layout.types},
            "layout": layout.to_json(),
            "source_stream": rel + ".raw",
            "description": description,
            **extra,
        }
        blobs[name] = raw
        parsed[name] = records

    for b in BUILDS:
        add(f"tables_{b}", f"{b}/tables", b, "tables",
            "FAST_HUCMINIT(12 s) then every table as READ (record 'read') and the derived "
            "tables (record 'derived': ima, chucm, both mg mass grids, ECOALMASSM, BRKWEIGHT, "
            "Prob, Gain_Var_New, NND, ikr_spon_break, aerosol spectra and grids, DROPRADII)")
        add(f"kernels_{b}", f"{b}/kernels", b, "kernels",
            "cwll and every cw* kernel family after Kernals_KS(12 s, 33, p) at 1000, 950, "
            "850, 750, 700, 600, 500, 400, 300 and 250 hPa")
        for f in SWEEPS:
            add(f"sweep_{f}_{b}", f"{b}/sweep_{f}", b, "sweep", SWEEP_TEXT[f])
        add(f"kc1_summary_{b}", f"{b}/kc1_summary", b, "kc1",
            "KC-1 initial state, a trajectory record after every one of the 300 calls, and "
            "the run's acceptance record with the branch counts")
        add(f"kc1_replay_{b}", f"{b}/kc1_replay", b, "kc1",
            "KC-1 full pre-call (after advection) and post-call state at the 16 replay steps")
        add(f"kc1_probe_{b}", f"{b}/kc1_probe", b, "kc1",
            "KC-1 probe records inside FAST_SBM and COAL_BOTT_NEW at the probe steps and "
            "levels, cwll for the first 40 collision calls recorded, and the branch counts")
    add("kc1_case", "o64/kc1_case", "o64", "kc1",
        "KC-1 case definition in float64 (grid, sounding, forcing, rain spectrum, replay "
        "and probe steps) and the literal flux-form advection check")
    add("kc1_crossreplay_o64_on_o32", "o64/kc1_crossreplay", "o64", "kc1",
        "the promoted build's FAST_SBM, once per replay step, on the native build's "
        "pre-call states (O64x in the native envelope gate)", input_states="o32")
    add("kc1_sensitivity_o64", "o64/kc1_sensitivity", "o64", "kc1",
        "KC-1 rerun in the promoted build with every initial theta moved up by one ULP: "
        "the Fortran's own self-divergence curve")
    m1_names = sorted(files)

    # Milestone 2 families -----------------------------------------------------
    for b in BUILDS:
        for f in COLD_SWEEPS:
            add(f"sweep_{f}_{b}", f"{b}/sweep_{f}", b, "sweep", COLD_SWEEP_TEXT[f],
                milestone=2)
        add(f"sweep_coal_cold_graupel_{b}", f"{b}/sweep_coal_cold_graupel", b, "sweep",
            COLD_SWEEP_TEXT["coal_cold_graupel"], milestone=2, graupel_build=True)
        add(f"kc2_summary_{b}", f"{b}/kc2_summary", b, "kc2",
            "KC-2 initial state, a record after every one of the 300 calls (surface "
            "precipitation and minimum temperature per column), profiles every 30 calls, and "
            "the acceptance record with the per-column counts", milestone=2)
        add(f"kc2_replay_{b}", f"{b}/kc2_replay", b, "kc2",
            "KC-2 full pre-call (after advection; the whole domain's theta, qv, th_old and "
            "qv_old, the tile's bins) and post-call state at the 7 replay steps", milestone=2)
        add(f"kc2_probe_{b}", f"{b}/kc2_probe", b, "kc2",
            "KC-2 probe records inside FAST_SBM, JERNUCL01_KS, ONECOND2, ONECOND3 and "
            "COAL_BOTT_NEW at 4 steps and 8 levels of the four cold columns, the pre-pass "
            "arrays of the whole tile, and the per-column counts", milestone=2)
        add(f"kc2g_replay_{b}", f"{b}/kc2g_replay", b, "kc2g",
            "KC-2 in the graupel build: full pre-call and post-call state at 3 steps",
            milestone=2, graupel_build=True)
        add(f"kc2g_summary_{b}", f"{b}/kc2g_summary", b, "kc2g",
            "KC-2 in the graupel build: a record after every call and the acceptance record "
            "with the per-column counts", milestone=2, graupel_build=True)
        add(f"kc2edge_replay_{b}", f"{b}/kc2edge_replay", b, "kc2edge",
            "the low-edge KC-2 run (tile i, j = 1..3, a Hallett-Mossop column in the corner): "
            "the case record, the static fields, and full pre-call and post-call state at "
            "steps 2, 4 and 6", milestone=2)
        add(f"kc2edge_probe_{b}", f"{b}/kc2edge_probe", b, "kc2edge",
            "the low-edge KC-2 run in the probe build: the pre-pass arrays of the tile "
            "(DSUPICE_XYZ, DTFREEZ_XYZ, the cloud base, and T_OLD, QV_OLD and pcgs over tile "
            "and halo) at steps 2, 4 and 6, and the per-column counts", milestone=2)
    add("kc2_case", "o64/kc2_case", "o64", "kc2",
        "KC-2 case definition in float64 (domain, tile, column types, grid, soundings, "
        "winds, forcing, initial bins, replay and probe steps)", milestone=2)
    add("kc2_crossreplay_o64_on_o32", "o64/kc2_crossreplay", "o64", "kc2",
        "the promoted build's FAST_SBM, once per replay step, on the native build's KC-2 "
        "pre-call states", input_states="o32", milestone=2)
    add("kc2_sensitivity_o64", "o64/kc2_sensitivity", "o64", "kc2",
        "KC-2 rerun in the promoted build with every initial theta one ULP higher: the "
        "Fortran's own self-divergence curve", milestone=2)

    # Refusals container ---------------------------------------------------
    members = []
    container = bytearray()
    for b in BUILDS:
        for n in range(1, 7):
            base = out / b / "refusals"
            raw, layout, records = read_stream(base, f"refusal_{n}")
            status = int((base / f"refusal_{n}.status").read_text().strip())
            err = (base / f"refusal_{n}.err").read_text(errors="replace")
            msg_path = base / f"refusal_{n}.message"
            message = msg_path.read_text().strip() if msg_path.exists() else ""
            if f"FSBM_REFUSAL_CALLING {n}" not in err:
                raise PackRefusal(f"refusal {n} ({b}) never reached its call")
            if "FSBM_REFUSAL_RETURNED" in err:
                raise PackRefusal(f"refusal {n} ({b}): the routine returned")
            if status == 99:
                if not message or f"WRF_ERROR_FATAL: {message}" not in err:
                    raise PackRefusal(f"refusal {n} ({b}): exit 99 without its message")
            elif status == 0:
                if message:
                    raise PackRefusal(f"refusal {n} ({b}): exit 0 with a fatal message")
            else:
                raise PackRefusal(f"refusal {n} ({b}): exit status {status}")
            routine, what = REFUSAL_TEXT[n]
            members.append({
                "case": n, "build": b, "routine": routine, "input": what,
                "exit_status": status, "message": message,
                "stop_kind": "wrf_error_fatal" if status == 99 else "bare STOP",
                "wrf_line": wrf_line_of(message, access_source) if message else 5360,
                "offset": len(container), "bytes": len(raw), "sha256": sha256(raw),
                "layout": layout.to_json(),
            })
            container += raw
    container = bytes(container)
    cold_container, cold_entry = refusal_container(
        out, "refusal_cold", "refusals_cold", REFUSAL_COLD, access_source,
        lambda n, b: REFUSAL_COLD[n][2][b])
    cold_entry["description"] = ("cold-path inputs that reach a WRF stop, one process per "
                                 "case: the input records written before the call, the "
                                 "message and exit status, and whether FAST_SBM can reach "
                                 "the input")
    cold_entry["milestone"] = 2
    files["refusals_cold"] = cold_entry
    blobs["refusals_cold"] = cold_container
    files["refusals"] = {
        "file": "refusals.bin.gz", "family": "refusals", "build": "o64+o32",
        "container": "refusals",
        "format": "concatenated record streams, one per case and build; each member's "
                  "offset, length, SHA-256 and layout are listed",
        "dtype": "record streams",
        "bytes": len(container), "sha256": sha256(container),
        "record_count": sum(sum(t["count"] for t in m["layout"]) for m in members),
        "members": members,
        "description": "warm-path inputs that reach a WRF stop, one process per case: the "
                       "input records written before the call, the message and exit status",
    }
    blobs["refusals"] = container

    # Controls ---------------------------------------------------------------
    for b in BUILDS:
        for key in (f"determinism_pristine_{b}_control", f"inert_access_{b}_control",
                    f"inert_probe_{b}_control", f"inert_probe_run2_{b}_control"):
            if controls.get(key) is not True:
                raise PackRefusal(f"control {key} is {controls.get(key)}")
    bad = [k for k, v in controls.items()
           if k.startswith(("determinism_", "inert_")) and v is not True]
    if bad:
        raise PackRefusal(f"controls not true: {bad}")
    control_hashes = dict(
        (line.split()[1], line.split()[0])
        for line in (out / "control_sha256.txt").read_text().splitlines() if line.strip())

    selfreplay = {}
    for b in BUILDS:
        _, _, sr = read_stream(out, f"{b}/kc1_selfreplay")
        rep_post = records_of(parsed[f"kc1_replay_{b}"], "post")
        sr_post = records_of(sr, "post")
        same = (len(rep_post) == len(sr_post) == 16
                and all(fields_equal(a, c) for a, c in zip(rep_post, sr_post)))
        if not same:
            raise PackRefusal(f"self-replay control failed in {b}: FAST_SBM carries state "
                              "between calls that the replay records do not hold")
        selfreplay[f"selfreplay_{b}_identical"] = True
        for cm in ("kc2", "kc2g"):
            _, _, sr = read_stream(out, f"{b}/{cm}_selfreplay")
            rep_post = records_of(parsed[f"{cm}_replay_{b}"], "post")
            sr_post = records_of(sr, "post")
            same = (len(rep_post) == len(sr_post) > 0
                    and all(fields_equal(a, c) for a, c in zip(rep_post, sr_post)))
            if not same:
                raise PackRefusal(f"{cm} self-replay control failed in {b}")
            selfreplay[f"selfreplay_{cm}_{b}_identical"] = True
        if controls.get(f"graupel_build_differs_from_hail_{b}") is not True:
            raise PackRefusal(f"the graupel build gave the hail build's KC-2 answer in {b}")
    cross2 = records_of(parsed["kc2_crossreplay_o64_on_o32"], "post")
    if [int(r["step"]) for r in cross2] != [int(r["step"]) for r in
                                            records_of(parsed["kc2_replay_o32"], "post")]:
        raise PackRefusal("KC-2 cross-replay steps differ from the replay steps")

    cross = records_of(parsed["kc1_crossreplay_o64_on_o32"], "post")
    if [int(r["step"]) for r in cross] != [int(r["step"]) for r in
                                           records_of(parsed["kc1_replay_o32"], "post")]:
        raise PackRefusal("cross-replay steps differ from the replay steps")

    xl_checks = {}
    for b in BUILDS:
        derived = records_of(parsed[f"tables_{b}"], "derived")[0]
        kern = records_of(parsed[f"kc1_probe_{b}"], "kernel")
        if not kern:
            raise PackRefusal(f"kc1_probe_{b} holds no kernel record")
        same = all(np.array_equal(k["xl_mg"], derived["xl_mg_6078"]) for k in kern)
        if not same:
            raise PackRefusal(f"{b}: the module's xl_mg differs from the :6078 expression")
        xl_checks[f"xl_mg_module_equals_6078_expression_{b}"] = True

    # KC-1 case in both builds -------------------------------------------------
    _, _, case32 = read_stream(out, "o32/kc1_case")
    case64 = parsed["kc1_case"]
    differing = []
    for r64, r32 in zip(case64, case32):
        for key in r64.fields:
            if not np.array_equal(np.asarray(r64[key]), np.asarray(r32[key])):
                differing.append(f"{r64.type}.{key}")
    case_rec = records_of(case64, "case")[0]
    lit = records_of(case64, "literal_advection_check")[0]

    # Acceptance ---------------------------------------------------------------
    acceptance = {}
    for b in BUILDS:
        acc = records_of(parsed[f"kc1_summary_{b}"], "acceptance")[0]
        counts = {k: [int(x) for x in np.asarray(acc[k])] for k in acc.fields
                  if np.asarray(acc[k]).dtype.kind == "i"}
        tmin = [float(x) for x in np.minimum(acc["tmin_run_pre"], acc["tmin_run_post"])]
        rain = [float(x) for x in acc["rainnc_final"]]
        checks = {
            "min_temperature_above_275.15_every_column": all(t > MIN_TEMPERATURE_K for t in tmin),
            "surface_rain_c1": rain[0] > 0 and counts["rain_steps"][0] > 0,
            "surface_rain_c2": rain[1] > 0 and counts["rain_steps"][1] > 0,
            "surface_rain_c3": rain[2] > 0 and counts["rain_steps"][2] > 0,
            "collisional_breakup_c3": counts["coal_breakup"][2] > 0,
            "cloud_base_activation": sum(counts["nucl_cb_act"]) > 0,
            "in_cloud_activation": sum(counts["nucl_ic_act"]) > 0,
            "onecond1_evaporation_branch": sum(counts["cond1_evap"]) > 0,
            "spontaneous_breakup": sum(counts["spont_taken"]) > 0,
        }
        if not all(checks.values()):
            raise PackRefusal(f"KC-1 acceptance fails in {b}: "
                              f"{[k for k, v in checks.items() if not v]}")
        acceptance[b] = {
            "columns": list(COLUMN_NAMES),
            "min_temperature_k": tmin,
            "rainnc_final_mm": rain,
            "counts": counts,
            "checks": checks,
        }

    # KC-2 acceptance, hail and graupel builds -----------------------------------
    acceptance2, acceptance2g = {}, {}
    for b in BUILDS:
        for cm, target, graupel in (("kc2", acceptance2, False), ("kc2g", acceptance2g, True)):
            acc = kc2_acceptance(parsed[f"{cm}_summary_{b}"], graupel)
            if not all(acc["checks"].values()):
                raise PackRefusal(f"{cm} acceptance fails in {b}: "
                                  f"{[k for k, v in acc['checks'].items() if not v]}")
            target[b] = acc
    acceptance_edge = {}
    for b in BUILDS:
        acc = kc2edge_acceptance(parsed[f"kc2edge_replay_{b}"], parsed[f"kc2edge_probe_{b}"])
        if not all(acc["checks"].values()):
            raise PackRefusal(f"kc2edge acceptance fails in {b}: "
                              f"{[k for k, v in acc['checks'].items() if not v]}")
        acceptance_edge[b] = acc
    case2 = records_of(parsed["kc2_case"], "case")[0]
    _, _, case2_32 = read_stream(out, "o32/kc2_case")
    differing2 = []
    for r64, r32 in zip(parsed["kc2_case"], case2_32):
        for key in r64.fields:
            if not np.array_equal(np.asarray(r64[key]), np.asarray(r32[key])):
                differing2.append(f"{r64.type}.{key}")

    # Milestone 1 streams must be the ones already packed --------------------------
    m1_check = {}
    if previous is not None:
        changed = [n for n in m1_names if n in previous.get("files", {})
                   and previous["files"][n]["sha256"] != files[n]["sha256"]]
        missing_before = [n for n in m1_names if n not in previous.get("files", {})]
        if changed and not allow_m1_change:
            raise PackRefusal(f"milestone 1 streams changed: {changed}")
        m1_check = {"milestone1_streams_compared": len(m1_names) - len(missing_before),
                    "milestone1_streams_changed": changed,
                    "milestone1_streams_unchanged": not changed}

    # Evidence: O32 at WRF's production flags ----------------------------------
    _, _, wrfopt = read_stream(out, "o32wrf/kc1_summary")
    _, _, plain = read_stream(out, "o32wrf/pristine_kc1_summary")
    s_opt, s_plain = records_of(wrfopt, "step"), records_of(plain, "step")
    first_diff = next((int(a["step"]) for a, c in zip(s_opt, s_plain)
                       if not fields_equal(a, c)), None)
    _, _, wrfopt2 = read_stream(out, "o32wrf/kc2_summary")
    _, _, plain2 = read_stream(out, "o32wrf/pristine_kc2_summary")
    s_opt2, s_plain2 = records_of(wrfopt2, "step"), records_of(plain2, "step")
    evidence_kc2 = {
        "flags": build_info["flags"]["o32_wrf"],
        "gated": False,
        "control_identical_to_o32": bool(controls.get("wrfopt_o32_kc2_control_identical_to_o32")),
        "first_step_that_differs": next((int(a["step"]) for a, c in zip(s_opt2, s_plain2)
                                         if not fields_equal(a, c)), None),
        "max_abs_diff_over_run": {
            k: float(max(np.max(np.abs(np.asarray(a[k], dtype=np.float64)
                                       - np.asarray(c[k], dtype=np.float64)))
                         for a, c in zip(s_opt2, s_plain2)))
            for k in ("rainnc", "snownc", "graupelnc", "tmin")
        },
        "final_mm_o32": {k: [float(x) for x in np.asarray(s_plain2[-1][k]).ravel(order="F")]
                         for k in ("rainnc", "snownc", "graupelnc")},
        "final_mm_o32_wrf": {k: [float(x) for x in np.asarray(s_opt2[-1][k]).ravel(order="F")]
                             for k in ("rainnc", "snownc", "graupelnc")},
    }
    evidence = {
        "flags": build_info["flags"]["o32_wrf"],
        "gated": False,
        "control_identical_to_o32": bool(controls.get("wrfopt_o32_control_identical_to_o32")),
        "first_step_that_differs": first_diff,
        "rainnc_final_mm_o32": [float(x) for x in s_plain[-1]["rainnc"]],
        "rainnc_final_mm_o32_wrf": [float(x) for x in s_opt[-1]["rainnc"]],
        "max_abs_diff_over_run": {
            k: float(max(np.max(np.abs(np.asarray(a[k], dtype=np.float64)
                                       - np.asarray(c[k], dtype=np.float64)))
                         for a, c in zip(s_opt, s_plain)))
            for k in ("th", "qv", "qc", "qr", "qnc", "rainnc")
        },
    }

    # Transforms ---------------------------------------------------------------
    access_diff = (out / "transform_access.diff").read_text()
    probe_diff = (out / "transform_probe.diff").read_text()
    a_hunks, a_removed, a_added = diff_body(access_diff)
    p_hunks, p_removed, p_added = diff_body(probe_diff)
    if a_removed != [" PRIVATE"] or a_added or a_hunks != ["@@ -3862 +3861,0 @@"]:
        raise PackRefusal("the access transform diff is not the one deleted line 3862")
    call_re = re.compile(r"^\s*CALL fsbm_probe_[a-z0-9_]+\(.*\)$")
    if p_removed or not p_added or not all(call_re.match(x) for x in p_added):
        raise PackRefusal("the probe transform diff is not inserted CALL lines only")
    anchors = json.loads((out / "probe_anchors.json").read_text())
    graupel_tx = {}
    for key, fname in (("graupel", "transform_graupel"), ("graupel_probe",
                                                          "transform_graupel_probe")):
        gdiff = (out / f"{fname}.diff").read_text()
        g_hunks, g_removed, g_added = diff_body(gdiff)
        summary = json.loads((out / f"{fname}.json").read_text())
        if (g_removed != [" INTEGER,PARAMETER :: hail_opt = 1"]
                or g_added != [" INTEGER,PARAMETER :: hail_opt = 0"] or len(g_hunks) != 1):
            raise PackRefusal(f"the {key} transform diff is not the one hail_opt line")
        if key == "graupel" and g_hunks != ["@@ -3886 +3886 @@"]:
            raise PackRefusal("the graupel transform did not change line 3886")
        if key == "graupel_probe" and summary["input_sha256"] != anchors["output_sha256"]:
            raise PackRefusal("the graupel probe build is not the probe source plus one line")
        graupel_tx[key] = {**summary, "diff": gdiff}
    transforms = {
        "access": {**json.loads((out / "transform_access.json").read_text()),
                   "diff": access_diff},
        "probe": {"output_sha256": anchors["output_sha256"],
                  "inserted_lines": anchors["inserted_lines"],
                  "removed_lines": anchors["removed_lines"],
                  "anchors": anchors["anchors"],
                  "diff": probe_diff},
        **graupel_tx,
    }

    # Write --------------------------------------------------------------------
    dest.mkdir(parents=True, exist_ok=True)
    # A stream whose bytes are unchanged keeps its compressed file as it is, so
    # the milestone 1 fixtures stay byte-identical on disk whatever zlib this
    # Python links.
    existing = {}
    for old in dest.glob("*.bin.gz"):
        existing[old.name] = old.read_bytes()
        old.unlink()
    total = 0
    for name in sorted(blobs):
        gz = existing.get(files[name]["file"])
        try:
            if gz is not None and gzip.decompress(gz) != blobs[name]:
                gz = None
        except (OSError, EOFError, zlib.error):
            gz = None
        if gz is None or gz[3] & 0x08 or gz[4:8] != bytes(4):
            gz = gzip_bytes(blobs[name])
        if gzip.decompress(gz) != blobs[name]:
            raise PackRefusal(f"{name}: gzip round trip failed")
        (dest / files[name]["file"]).write_bytes(gz)
        files[name]["gzip_bytes"] = len(gz)
        files[name]["gzip_sha256"] = sha256(gz)
        total += len(gz)
    if total > SIZE_LIMIT:
        raise PackRefusal(f"packed fixtures total {total} bytes, over {SIZE_LIMIT}")

    manifest = {
        "schema": "fsbm-oracle-manifest 1",
        "generated_by": "tools/fast_sbm_wrf480_oracle/pack_fixtures.py",
        "reader": "tools/fast_sbm_wrf480_oracle/fixtures.py",
        "scheme": "WRF v4.8.0 fast spectral bin microphysics (mp_physics 30, FSBM-2)",
        "compiler": build_info["compiler"],
        "build_host": {"host": build_info["host"], "cpu": build_info["cpu"],
                       "kernel": build_info["kernel"], "date_utc": build_info["date_utc"],
                       "wall_seconds": build_info["wall_seconds"]},
        "flags": build_info["flags"],
        "builds": {k: {"role": v, "flags": (build_info["flags"]["common"] + " "
                                             + build_info["flags"][k])}
                   for k, v in BUILD_ROLES.items()},
        "fast_sbm_compile_warnings": build_info["fast_sbm_compile_warnings"],
        "source_pins": pins,
        "harness_sha256": harness,
        "transforms": transforms,
        "controls": {**{k: v for k, v in controls.items()
                        if k.startswith(("determinism_", "inert_", "graupel_build_"))},
                     **selfreplay, **xl_checks, **m1_check,
                     "kc1_control_stream_sha256": control_hashes,
                     "kc2_control_stream_sha256": dict(
                         (line.split()[1], line.split()[0])
                         for line in (out / "control_sha256_kc2.txt").read_text().splitlines()
                         if line.strip())},
        "kc1": {
            "definition": kc1_definition(case_rec, lit),
            "case_fields_that_differ_between_builds": differing,
            "acceptance": acceptance,
        },
        "findings": FINDINGS,
        "kc2": {
            "definition": kc2_definition(case2),
            "case_fields_that_differ_between_builds": differing2,
            "acceptance": acceptance2,
            "acceptance_graupel": acceptance2g,
            "catalogue_counts": {"note": "sums over the tile of the counts the catalogue "
                                         "rows D11 to D16, S1 and S2 point at",
                                 **{f"hail_{b}": acceptance2[b]["catalogue_counts"]
                                    for b in BUILDS},
                                 **{f"graupel_{b}": acceptance2g[b]["catalogue_counts"]
                                    for b in BUILDS}},
            "counts": "per tile column, i fastest then j: " + ", ".join(KC2_COLUMNS),
            "live_cold_collision_calls": {"hail": list(HAIL_LIVE_CALLS),
                                          "graupel": list(GRAUPEL_LIVE_CALLS),
                                          "names": list(CC_NAMES)},
        },
        "kc2edge": {"definition": KC2EDGE_DEFINITION, "acceptance": acceptance_edge},
        "findings_cold": FINDINGS_COLD,
        "evidence_o32_wrf": evidence,
        "evidence_o32_wrf_kc2": evidence_kc2,
        "reproduction": (reproduction_record(out, repro, repro_label)
                         if repro is not None else None),
        "required_families": required_families(),
        "total_gzip_bytes": total,
        "size_limit_bytes": SIZE_LIMIT,
        "files": files,
    }
    missing = [n for n in manifest["required_families"] if n not in files]
    if missing:
        raise PackRefusal(f"required families missing: {missing}")
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8", newline="\n")
    (dest / "MANIFEST.md").write_text(manifest_md(manifest), encoding="utf-8", newline="\n")
    return manifest


def required_families() -> list[str]:
    names = ["refusals", "kc1_case", "kc1_crossreplay_o64_on_o32", "kc1_sensitivity_o64"]
    for b in BUILDS:
        names += [f"tables_{b}", f"kernels_{b}", f"kc1_summary_{b}", f"kc1_replay_{b}",
                  f"kc1_probe_{b}"]
        names += [f"sweep_{f}_{b}" for f in SWEEPS]
    names += required_families_m2()
    return sorted(names)


def required_families_m2() -> list[str]:
    names = ["refusals_cold", "kc2_case", "kc2_crossreplay_o64_on_o32", "kc2_sensitivity_o64"]
    for b in BUILDS:
        names += [f"kc2_summary_{b}", f"kc2_replay_{b}", f"kc2_probe_{b}",
                  f"kc2g_replay_{b}", f"kc2g_summary_{b}", f"sweep_coal_cold_graupel_{b}",
                  f"kc2edge_replay_{b}", f"kc2edge_probe_{b}"]
        names += [f"sweep_{f}_{b}" for f in COLD_SWEEPS]
    return sorted(names)


def kc1_definition(case: fx.Record, lit: fx.Record) -> dict:
    return {
        "tile": "four columns called together in one FAST_SBM call, i = 1..4, j = 1, "
                "ids..ide = 1..5, jds..jde = 1..2, memory i 0..5, j 0..2",
        "columns": {
            "c1": "land (XLAND 1, continental aerosol), surface 1000 hPa, 303 K; updraft "
                  "w1 sin(pi z / H) with w1 = 2 m/s while t < 900 s, then still",
            "c2": "water (XLAND 2, maritime), surface 1005 hPa, 303 K; forcing as c1",
            "c3": "water, surface 995 hPa, 303 K; still air; Marshall-Palmer rain for "
                  "20 mm/h in bins 15-33 above 1.2 km; RH 60 percent below 1.2 km",
            "c4": "land, surface 850 hPa, 296 K, last in the tile; w1 = 0.3 m/s while "
                  "t < 1800 s; RH 99 percent at and above 2.8 km (the moist cap)",
        },
        "grid": "40 levels, lowest layer 50 m, each layer 2.5 percent deeper than the one "
                "below, top 3369 m; w on the 41 full levels, zero at the bottom and top",
        "sounding": "T = Tsfc - 5.5 K/km, p hydrostatic for that lapse rate with "
                    "g = 9.81, Rd = 287, cp = 1004.5, Exner (p/1e5)^(Rd/cp); RH 92 percent "
                    "below 800 m, 80 percent to 1.8 km, 55 percent above (c3 and c4 as "
                    "stated); qv from es = 2.53e12 exp(-5420/T) dyn/cm2",
        "moist_cap": {"column": int(case["rh_cap_column"]),
                      "height_m": float(case["rh_cap_height"]),
                      "relative_humidity": float(case["rh_cap_value"]),
                      "reason": "finding D9: the cloud-base activation branch is reachable "
                                "only at a level whose w is at most 10 cm/s or that is the "
                                "highest unsaturated one; c4's top level is such a level"},
        "time_step_s": float(case["dt"]),
        "steps": int(case["nsteps"]),
        "advection": "between calls theta, qv and all 142 advected bins (the WRF chem "
                     "form) by first-order upwind with fixed density and zero flux at the "
                     "bottom and top: g_k = 0.5 (rho_k + rho_k+1) w_k+1 on face k, "
                     "up_k = phi_k if g_k >= 0 else phi_k+1, "
                     "phi_k += dt * (-((g_k (up_k - phi_k)) - (g_k-1 (up_k-1 - phi_k))) "
                     "/ (rho_k dz_k)), evaluated in that order in the build's default real "
                     "(the flux form with its divergence term restored); th_old and qv_old "
                     "are not advected; w at call n is the forcing at t = (n - 1) dt and "
                     "the advection before call n uses it",
        "literal_flux_form_check": {
            "steps": int(lit["steps"]),
            "c1_min_temperature_plain_flux_form_k": float(lit["tmin_plain_flux_form"]),
            "c1_min_temperature_case_operator_k": float(lit["tmin_case_operator"]),
            "note": "the plain flux form the design names empties the lowest level under "
                    "a fixed density and a height-dependent w; the case uses the form above",
        },
        "replay_steps": [int(x) for x in case["replay_steps"]],
        "probe_steps": [int(x) for x in case["probe_steps"]],
        "probe_levels": [int(x) for x in case["probe_levels"]],
        "call": "FAST_SBM(itimestep = n, dt = 12, dx = dy = 1000, sbm_diagnostics = 0, "
                "diagflag absent, num_sbmradar = 1) after FAST_HUCMINIT(12)",
    }


FINDINGS = {
    "D9_cloud_base_eqv_precedence": (
        "FAST_SBM's cloud-base test at :4387 joins its terms with .and. around "
        "'K_found .eqv. .FALSE.'; .EQV. binds more loosely than .AND., so the test is "
        "(SUP_WATER > 0 .and. w > 10 cm/s .and. K_found) .eqv. (.FALSE. .and. ...), whose "
        "right side is always false. KZ_Cloud_Base therefore ends as the highest level that "
        "is not (supersaturated with w above 10 cm/s), or 1, instead of the lowest level "
        "above the second that is. With the design's sounding the top level never "
        "saturates, so KZ is level 40 in every column at every call (the probe records "
        "show it) and the cloud-base branch never ran in 300 steps; with the c4 moist cap "
        "it runs once, at step 40, level 40, column c4, in both builds."),
    "still_air_has_no_diffusional_growth": (
        "Condensation and evaporation are driven only by the change of T and qv since the "
        "previous call (DIFFU = 0 when DIV1 equals DIV3, :4615). In a column with no "
        "advection since the last call (c3, still air) no drop grows or evaporates, even "
        "rain falling into 60 percent humidity: ONECOND1 is called 0 times in c3."),
    "spontaneous_breakup_bin": (
        "Spontanous_Init sets ikr_spon_break to the first bin with radius at least 0.3 cm, "
        "which on the 33-bin grid is bin 33 (3.25 mm). Spont_Rain_BreakUp does work only "
        "when bin 33 holds any positive value; in the float64 build that is almost every "
        "rainy call, in the float32 build noticeably fewer (the counts differ by build)."),
    "falflux_bare_stop": (
        "FALFLUXHUCM_Z stops the run with a bare STOP when every fall speed is zero "
        "(:5360): the process exits with status 0 and no message (refusal case 6)."),
}


def manifest_md(m: dict) -> str:
    acc = m["kc1"]["acceptance"]
    lines = [
        "# Fast SBM Fortran oracle fixtures",
        "",
        "These files are the reference outputs of WRF v4.8.0's fast spectral bin "
        "microphysics (mp_physics 30), compiled from the pinned WRF source and run on the "
        "build node. The float64 host port in `woof/core/fast_sbm/` is gated against them. "
        "Read them only through `tools/fast_sbm_wrf480_oracle/fixtures.py`, which checks "
        "each file's SHA-256 before it returns anything. `MANIFEST.json` holds every number "
        "quoted here.",
        "",
        "## How they were made",
        "",
        f"Compiler: {m['compiler']} on {m['build_host']['host']} "
        f"({m['build_host']['cpu']}), {m['build_host']['date_utc']}, CPU only. The whole "
        f"harness ran in {m['build_host']['wall_seconds']} seconds.",
        "",
        "Two builds are packed. O64 promotes every default real to float64 "
        f"(`{m['builds']['o64']['flags']}`) and is the tight gate for the float64 host. O32 "
        f"is WRF as it runs (`{m['builds']['o32']['flags']}`) and is compared through an "
        "envelope. A third build at WRF's production optimisation "
        f"(`{m['flags']['o32_wrf']}`) is evidence only and is never gated.",
        "",
        "The WRF sources were LF-normalized and refused unless their SHA-256 matched the "
        "pins: `phys/module_mp_fast_sbm.F` "
        f"{m['source_pins']['sources']['phys/module_mp_fast_sbm.F']['lf_sha256']} and "
        "`phys/module_mp_SBM_polar_radar.F` "
        f"{m['source_pins']['sources']['phys/module_mp_SBM_polar_radar.F']['lf_sha256']}. "
        f"The table archive {m['source_pins']['archive']['file']} "
        f"({m['source_pins']['archive']['bytes']} bytes, "
        f"{m['source_pins']['archive']['sha256']}) and its 18 members were checked the same "
        "way.",
        "",
        "Two source transforms let the runners reach private routines and record inside "
        "FAST_SBM. The access transform deletes exactly one line, line 3862 (` PRIVATE` of "
        "module module_mp_fast_sbm). The probe transform adds "
        f"{m['transforms']['probe']['inserted_lines']} CALL lines and nothing else. The "
        "controls prove neither changes the answer: over the whole KC-1 run the pristine, "
        "access-transformed and probe builds give byte-identical FAST_SBM outputs in each "
        "precision, two runs of the same build are byte-identical, and each build replayed "
        "on its own recorded pre-call states in a fresh process gives back its own "
        "post-call states byte for byte, so FAST_SBM carries no state between calls that "
        "the replay records miss.",
        "",
        "## What is in them",
        "",
        "Every file is one record stream: each record is an int32 type tag followed by that "
        "type's fields, raw little-endian, arrays in Fortran order. O64 files hold only "
        "float64 and int32 fields; O32 files hold float32 where WRF declares default real "
        "and float64 where WRF declares REAL(KIND=8).",
        "",
        "* `tables_*`: every table as read from `SBM_input_33` and the derived tables after "
        "FAST_HUCMINIT with a 12 s step.",
        "* `kernels_*`: the collision kernels after Kernals_KS at ten pressures from 1000 "
        "to 250 hPa.",
        "* `sweep_<routine>_*`: one family per process routine, inputs exactly as passed "
        "and the outputs the routine returned.",
        "* `refusals`: six inputs that stop WRF, the input records, the message and the exit "
        "status, for both builds.",
        "* `kc1_case`: the warm kinematic column case, defined in float64.",
        "* `kc1_summary_*`: the state after every call; `kc1_replay_*`: full pre-call and "
        "post-call states at 16 steps; `kc1_probe_*`: records inside FAST_SBM and "
        "COAL_BOTT_NEW at 6 steps and 8 levels per column.",
        "* `kc1_crossreplay_o64_on_o32`: the O64 build run once on each O32 pre-call state.",
        "* `kc1_sensitivity_o64`: KC-1 again with every initial theta one ULP higher.",
        "",
        "## The warm column KC-1",
        "",
        "Four columns are called together, so the cross-column behaviour of FAST_SBM is "
        "inside the fixture. c1 is a land column with a 2 m/s updraft for 15 minutes, c2 "
        "the same over water, c3 still air with Marshall-Palmer rain above 1.2 km, and c4 "
        "a weak 0.3 m/s updraft at 850 hPa, last in the tile. The step is 12 s and the run "
        "is 300 steps (one hour). Between calls theta, qv and every bin are advected by a "
        "first-order upwind scheme.",
        "",
        "Two choices differ from the design text and are recorded. The advection is the "
        "flux form with its divergence term restored, because the plain flux form with a "
        "fixed density empties the lowest level (it cools c1's lowest level to "
        f"{m['kc1']['definition']['literal_flux_form_check']['c1_min_temperature_plain_flux_form_k']:.1f} K "
        "in 15 minutes). And c4 carries a moist cap, 99 percent relative humidity above "
        "2.8 km, because of the cloud-base finding below: without it the cloud-base "
        "activation branch is never reached.",
        "",
        "Acceptance, both builds (columns c1 to c4):",
        "",
    ]
    for b in BUILDS:
        a = acc[b]
        c = a["counts"]
        lines += [
            f"* {b.upper()}: lowest temperature "
            + ", ".join(f"{t:.2f} K" for t in a["min_temperature_k"])
            + "; surface rain " + ", ".join(f"{r:.3g} mm" for r in a["rainnc_final_mm"])
            + f"; cloud-base activations {c['nucl_cb_act']}, in-cloud activations "
            f"{c['nucl_ic_act']}; ONECOND1 evaporation calls {c['cond1_evap']}; collisional "
            f"breakup {c['coal_breakup']}; spontaneous breakup {c['spont_taken']}.",
        ]
    ev = m["evidence_o32_wrf"]
    lines += [
        "",
        "## Findings from the oracle",
        "",
    ]
    for key, text in m["findings"].items():
        lines.append(f"* {key}: {text}")
    lines += [
        "",
        "## Evidence only: WRF's production flags",
        "",
        "The native build at WRF's production optimisation is not byte-identical to the "
        f"-O0 native build (first difference at step {ev['first_step_that_differs']}). "
        "Surface rain after one hour: "
        + ", ".join(f"{x:.4f}" for x in ev["rainnc_final_mm_o32_wrf"]) + " mm against "
        + ", ".join(f"{x:.4f}" for x in ev["rainnc_final_mm_o32"])
        + " mm. This is recorded, never gated.",
        "",
    ]
    lines += manifest_md_m2(m)
    rp = m.get("reproduction")
    if rp:
        lines += [
            "## Reproduction on a second compiler",
            "",
            f"The same harness was run under {rp['label']} with {rp['compiler']}. "
            f"{rp['streams_identical']} of {rp['streams_compared']} raw streams, including "
            "the production-flag evidence build, came out byte-identical to the test-machine-1 "
            "streams"
            + (". " if not rp["streams_differing"] else
               f"; these differ: {', '.join(rp['streams_differing'])}. ")
            + "This is recorded, not gated.",
            "",
        ]
    lines += [
        f"Total packed size: {m['total_gzip_bytes']} bytes (limit {m['size_limit_bytes']}).",
        "",
        "WRF is public domain software from NCAR, operated by UCAR. The scheme is Alexander "
        "Khain's Hebrew University Cloud Model bin microphysics; FSBM-2 in WRF was coded and "
        "is maintained by Jacob Shpund (Shpund et al., 2019). The tables are NCAR's "
        "SBM_input_33 distribution.",
        "",
    ]
    return "\n".join(lines)


def manifest_md_m2(m: dict) -> list[str]:
    k2 = m["kc2"]
    lines = [
        "## Milestone 2: the cold half",
        "",
        "The same harness adds the ice processes. Twelve cold sweep families run in the "
        "probe build of each precision: FREEZ, J_W_MELT, ICE_NUCL (reached through "
        "JERNUCL01_KS, since ICE_NUCL is private to its module; the probe build records "
        "its own call), JERNUCL01_KS with ice, the ice branches of JERRATE_KS, "
        "JERTIMESC_KS and JERDFUN_KS, ONECOND2, ONECOND3 (with the JERSUPSAT_KS call "
        "inside each read back), coll_xyx_lwf, coll_xyy_lwf and coll_xyz_lwf, modkrn_KS "
        "CASE 11 (with INTERPOL_SE), and cold COAL_BOTT_NEW, which also runs in the "
        "graupel build. The 143 live cases of the 09-13/14 port carry over with their "
        "grids: FREEZ, J_W_MELT, ICE_NUCL, JERNUCL01_KS and modkrn_KS on the real tables, "
        "and its 24 collision cases, with their random non-symmetric kernels, three ways: "
        "on their synthetic grid with y = 1.25 x (the orientation cases, because WRF's "
        "five mass grids are equal and a swapped x and y cannot show on them), on the "
        "synthetic y = x, and on the real tables (x = y = xl_mg with the module's chucm "
        "and ima). Its draws came from numpy; here a Park-Miller generator started at the "
        "old seeds draws them, and every draw is in the records. In the cold COAL_BOTT_NEW "
        "records colleff is WRF's value only where colleff_defined is 1; elsewhere WRF "
        "returns an uninitialised value (D6 and -40 C) and the record holds 0.",
        "",
        "The graupel variant is built from a third transform that changes only line 3886, "
        "`hail_opt = 1` to `hail_opt = 0`, applied to the pristine source and to the "
        "probe source; each is refused unless that line is its whole diff. "
        f"The probe transform now adds {m['transforms']['probe']['inserted_lines']} CALL "
        "lines: milestone 1's and the cold anchors in FAST_SBM, JERNUCL01_KS, ONECOND2, "
        "ONECOND3 and COAL_BOTT_NEW.",
        "",
        "Every milestone 1 stream came out of the extended harness byte for byte as "
        "packed before"
        + (f" ({m['controls'].get('milestone1_streams_compared')} streams compared)."
           if m["controls"].get("milestone1_streams_unchanged") else "."),
        "",
        "### The cold refusals",
        "",
    ]
    for mem in m["files"]["refusals_cold"]["members"]:
        if mem["build"] != "o64":
            continue
        o32 = next(x for x in m["files"]["refusals_cold"]["members"]
                   if x["case"] == mem["case"] and x["build"] == "o32")
        lines.append(
            f"* Case {mem['case']}, {mem['routine']}: {mem['input']}. O64: {mem['stop_kind']}"
            + (f" at :{mem['wrf_line']}" if mem["wrf_line"] else "")
            + f"; O32: {o32['stop_kind']}. "
            + ("FAST_SBM can reach this input." if mem["reachable_from_fast_sbm"]
               else "FAST_SBM cannot reach this input."))
    d = k2["definition"]
    lines += [
        "",
        "### The cold column KC-2",
        "",
        f"{d['tile'][0].upper()}{d['tile'][1:]}. Four tile columns carry the cold physics:",
        "",
    ]
    for name, text in d["columns"].items():
        lines.append(f"* {name}: {text}.")
    lines += [
        "",
        f"{d['grid'][0].upper()}{d['grid'][1:]}. The step is 12 s and the run 300 steps. "
        f"The winds: {d['winds']}. Replay steps "
        f"{', '.join(str(x) for x in d['replay_steps'])}; probe steps "
        f"{', '.join(str(x) for x in d['probe_steps'])} on levels "
        f"{', '.join(str(x) for x in d['probe_levels'])} of the four cold columns. The "
        "graupel build runs the same case with a replay at steps 2, 150 and 300.",
        "",
        "Acceptance, asserted in both precisions (sums over the tile):",
        "",
    ]
    for b in BUILDS:
        a = k2["acceptance"][b]
        c = a["counts"]
        ag = k2["acceptance_graupel"][b]
        cg = ag["counts"]
        lines.append(
            f"* {b.upper()}: temperatures from {min(a['min_temperature_k']):.1f} K to "
            f"{max(a['max_temperature_k']):.1f} K; ICE_NUCL called in the six bands "
            + ", ".join(str(sum(c[f'icenucl_band{i}'])) for i in range(1, 7))
            + f" times; homogeneous freezing {sum(c['homfrz'])}; ONECOND1, 2 and 3 "
            f"{sum(c['cond1_calls'])}, {sum(c['cond2_calls'])} and {sum(c['cond3_calls'])} "
            f"calls; KCOND 30, 31 and 32 {sum(c['cond3_kcond30'])}, {sum(c['cond3_kcond31'])} "
            f"and {sum(c['cond3_kcond32'])}; Hallett-Mossop splinters {sum(c['hm_splinters'])}; "
            f"FREEZ to plates {sum(c['freez_to_plates'])} and to hail {sum(c['freez_to_hail'])}; "
            f"D11 {sum(c['d11_levels'])} levels; surface snow up to "
            f"{max(a['snownc_final_mm']):.3g} mm and hail up to "
            f"{max(a['graupelnc_final_mm']):.3g} mm. Graupel build: surface graupel up to "
            f"{max(ag['graupelnc_final_mm']):.3g} mm, Hallett-Mossop "
            f"{sum(cg['hm_splinters'])}.")
    ke = m["kc2edge"]
    lines += [
        "",
        "The low-edge run: " + ke["definition"]["tile"] + ". " + ke["definition"]["columns"]
        + ". " + ke["definition"]["run"] + ". In both precisions the corner's pre-pass "
        "values are nonzero on the I = 1 and J = 1 branches (DSUPICE_XYZ at "
        + ", ".join(f"{ke['acceptance'][b]['dsupice_nonzero_points_i1']} ({b.upper()})"
                    for b in BUILDS)
        + " recorded points of the column i = 1) and ICE_NUCL runs in the corner.",
    ]
    lines += [
        "",
        "Every live cold collision call ran in both precisions: calls "
        + ", ".join(str(x) for x in k2["live_cold_collision_calls"]["hail"])
        + " in the hail build and "
        + ", ".join(str(x) for x in k2["live_cold_collision_calls"]["graupel"])
        + " in the graupel build (numbered in source order, :8452 to :8563); the others "
        "never ran.",
        "",
        "### Findings from the cold oracle",
        "",
    ]
    for key, text in m["findings_cold"].items():
        lines.append(f"* {key}: {text}")
    ev = m["evidence_o32_wrf_kc2"]
    lines += [
        "",
        "Evidence only: the native build at WRF's production flags "
        + ("gives KC-2 byte for byte." if ev["control_identical_to_o32"] else
           "does not give the -O0 native build's KC-2 state byte for byte; its per-call "
           "surface precipitation and minimum temperatures first differ at step "
           f"{ev['first_step_that_differs']}, and after one hour its surface rain differs by "
           f"at most {ev['max_abs_diff_over_run']['rainnc']:.2g} mm. This is recorded, never "
           "gated."),
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the out/ directory build.sh wrote")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--reproduction", type=Path, default=None)
    parser.add_argument("--reproduction-label", default="")
    parser.add_argument("--allow-milestone1-change", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = pack(args.out, args.dest, args.reproduction, args.reproduction_label,
                        args.allow_milestone1_change)
    except (PackRefusal, fx.FixtureError) as exc:
        print(f"pack_fixtures: REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"packed {len(manifest['files'])} fixtures, {manifest['total_gzip_bytes']} bytes, "
          f"into {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
