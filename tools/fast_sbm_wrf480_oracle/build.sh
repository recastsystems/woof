#!/usr/bin/env bash
# Fast SBM oracle harness: build WRF v4.8.0 FAST_SBM three ways and two
# precisions (and the graupel variant two ways), run every runner, and collect
# the raw streams and control results that pack_fixtures.py turns into
# woof/data/fast_sbm/oracle/. Milestone 1 made the warm fixtures (KC-1 and
# the warm sweeps); milestone 2 adds the cold sweeps and refusals, the
# graupel builds, the cold column KC-2 and its low-edge tile, and leaves
# every milestone 1 stream as it was.
#
# usage: build.sh <wrf-tree> <work-dir> [--tables <SBM_input_33.tgz>] [--no-evidence]
#
#   <wrf-tree>  a directory holding phys/module_mp_fast_sbm.F and
#               phys/module_mp_SBM_polar_radar.F (a WRF checkout, or a copy of
#               just those two files under phys/); CRLF is accepted and
#               normalized to LF before the pins are checked
#   <work-dir>  private scratch; created, and everything goes under it
#   --tables    use a local copy of the NCAR archive instead of fetching it
#
# CPU only. The authority is gfortran 15.2.0 on the reference host; any other compiler is
# recorded as such in out/build_info.json. Every compile and run is niced.
# Output: <work-dir>/out/ (raw streams, layouts, controls.json,
# build_info.json, transform diffs and anchors).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -lt 2 ]; then
  sed -n '2,20p' "$0"
  exit 2
fi
WRF="$1"; WORK="$2"; shift 2
TGZ=""
EVIDENCE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --tables) TGZ="$2"; shift 2 ;;
    --no-evidence) EVIDENCE=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

NICE="nice -n 10"
RUN_TIMEOUT=600
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
SRC="$WORK/src"; OUT="$WORK/out"; TAB="$WORK/tables"
rm -rf "$SRC" "$OUT" "$WORK"/b_* "$WORK"/r_*
mkdir -p "$SRC" "$OUT" "$TAB"
T0=$(date +%s)

refuse() { echo "build.sh: REFUSED: $*" >&2; exit 3; }
pin() { awk -v k="$1" -v p="$2" '$1==k && $4==p {print $2}' "$HERE/sources.sha256"; }
pin_bytes() { awk -v k="$1" -v p="$2" '$1==k && $4==p {print $3}' "$HERE/sources.sha256"; }

# 1. Sources: LF-normalize and check the pins --------------------------------
for f in phys/module_mp_fast_sbm.F phys/module_mp_SBM_polar_radar.F; do
  [ -f "$WRF/$f" ] || refuse "missing $WRF/$f"
  tr -d '\r' < "$WRF/$f" > "$SRC/$(basename "$f")"
  got=$(sha256sum "$SRC/$(basename "$f")" | cut -c1-64)
  want=$(pin source "$f")
  [ "$got" = "$want" ] || refuse "$f LF SHA-256 $got is not the pin $want"
done

# 2. Tables: fetch (or take) the archive and check archive and member pins ---
URL=$(awk '$1=="url" {print $2}' "$HERE/sources.sha256")
if [ -n "$TGZ" ]; then
  cp "$TGZ" "$TAB/SBM_input_33.tgz"
else
  curl -fsSL --retry 3 -o "$TAB/SBM_input_33.tgz" "$URL"
fi
got=$(sha256sum "$TAB/SBM_input_33.tgz" | cut -c1-64)
[ "$got" = "$(pin archive SBM_input_33.tgz)" ] || refuse "archive SHA-256 $got is not the pin"
[ "$(stat -c %s "$TAB/SBM_input_33.tgz")" = "$(pin_bytes archive SBM_input_33.tgz)" ] || refuse "archive size is not the pin"
tar -xzf "$TAB/SBM_input_33.tgz" -C "$TAB"
while read -r kind sha bytes path; do
  [ "$kind" = member ] || continue
  f="$TAB/$path"
  [ -f "$f" ] || refuse "archive member $path missing"
  got=$(sha256sum "$f" | cut -c1-64)
  [ "$got" = "$sha" ] || refuse "member $path SHA-256 $got is not the pin"
  [ "$(stat -c %s "$f")" = "$bytes" ] || refuse "member $path size is not the pin"
done < "$HERE/sources.sha256"

# 3. Transforms ---------------------------------------------------------------
python3 -B "$HERE/transform_access.py" --in "$SRC/module_mp_fast_sbm.F" \
  --out "$SRC/module_mp_fast_sbm_access.F" --diff "$OUT/transform_access.diff" \
  > "$OUT/transform_access.json"
python3 -B "$HERE/instrument_probes.py" --in "$SRC/module_mp_fast_sbm.F" \
  --out "$SRC/module_mp_fast_sbm_probe.F" --diff "$OUT/transform_probe.diff" \
  --anchors "$OUT/probe_anchors.json" > /dev/null
# The graupel variant: hail_opt = 1 becomes 0 (line 3886), on the pristine
# source and on the probe source, each refused unless that is its one change.
python3 -B "$HERE/transform_graupel.py" --in "$SRC/module_mp_fast_sbm.F" --variant pristine \
  --out "$SRC/module_mp_fast_sbm_graupel.F" --diff "$OUT/transform_graupel.diff" \
  > "$OUT/transform_graupel.json"
PROBE_SHA=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_sha256"])' \
  "$OUT/probe_anchors.json")
python3 -B "$HERE/transform_graupel.py" --in "$SRC/module_mp_fast_sbm_probe.F" --variant probe \
  --expect-sha256 "$PROBE_SHA" --out "$SRC/module_mp_fast_sbm_graupel_probe.F" \
  --diff "$OUT/transform_graupel_probe.diff" > "$OUT/transform_graupel_probe.json"

# 4. Builds -------------------------------------------------------------------
COMMON="-cpp -DBUILD_SBM_FAST=1 -DIWORDSIZE=4 -ffree-form -ffree-line-length-none"
O64="-O0 -ffp-contract=off -freal-4-real-8"
O32="-O0 -ffp-contract=off"
OWRF="-O2 -ftree-vectorize -funroll-loops"
IOFLAGS="-O0 -ffp-contract=off"

build() {  # build <name> <source-variant> <flags> <runners...>
  local name="$1" variant="$2" flags="$3"; shift 3
  local d="$WORK/b_$name"
  mkdir -p "$d"
  (
    cd "$d"
    local src="$SRC/module_mp_fast_sbm.F"
    [ "$variant" = access ] && src="$SRC/module_mp_fast_sbm_access.F"
    [ "$variant" = probe ] && src="$SRC/module_mp_fast_sbm_probe.F"
    [ "$variant" = gpristine ] && src="$SRC/module_mp_fast_sbm_graupel.F"
    [ "$variant" = gprobe ] && src="$SRC/module_mp_fast_sbm_graupel_probe.F"
    $NICE gfortran $COMMON $IOFLAGS -DFSBM_PART_IO -c "$HERE/fsbm_probe.F90" -o fsbm_io.o
    $NICE gfortran $COMMON $flags -DFSBM_PART_PROBE -c "$HERE/fsbm_probe.F90" -o fsbm_probe.o
    $NICE gfortran $COMMON $flags -c "$HERE/stubs_wrf.F90" -o stubs.o
    $NICE gfortran $COMMON $flags -c "$SRC/module_mp_SBM_polar_radar.F" -o polar.o
    $NICE gfortran $COMMON $flags -c "$src" -o fast_sbm.o > fast_sbm.log 2>&1 || { cat fast_sbm.log; exit 1; }
    for r in "$@"; do
      $NICE gfortran $COMMON $flags -c "$HERE/$r.F90" -o "$r.o"
      $NICE gfortran -o "$r" "$r.o" fast_sbm.o polar.o stubs.o fsbm_probe.o fsbm_io.o
    done
    grep -ci "warning" fast_sbm.log > warnings.count || true
  )
}

build pristine_o64 pristine "$O64" run_column run_kc2
build pristine_o32 pristine "$O32" run_column run_kc2
build access_o64 access "$O64" run_column run_dump_tables run_kc2
build access_o32 access "$O32" run_column run_dump_tables run_kc2
build probe_o64 probe "$O64" run_column run_sweeps run_kc2
build probe_o32 probe "$O32" run_column run_sweeps run_kc2
build gpristine_o64 gpristine "$O64" run_kc2
build gpristine_o32 gpristine "$O32" run_kc2
build gprobe_o64 gprobe "$O64" run_kc2 run_sweeps
build gprobe_o32 gprobe "$O32" run_kc2 run_sweeps
if [ "$EVIDENCE" = 1 ]; then
  build wrfopt_o32 pristine "$OWRF" run_column run_kc2
fi

# 5. Runs ---------------------------------------------------------------------
rundir() {  # rundir <name> -> prints a fresh run directory with the tables linked
  local d="$WORK/r_$1"
  rm -rf "$d"; mkdir -p "$d"
  ln -s "$TAB/SBM_input_33" "$d/SBM_input_33"
  echo "$d"
}
run() {  # run <rundir> <log> <cmd...>
  local d="$1" log="$2"; shift 2
  ( cd "$d" && timeout "$RUN_TIMEOUT" $NICE "$@" > "$log.out" 2> "$log.err" )
}

SWEEPS="polysvp jerrate jersupsat jerdfun jernewf onecond1 nucleation coll_xxx coal_bott breakup spont falflux"
for p in o64 o32; do
  # KC-1 in every build, twice for the pristine and probe builds
  for b in pristine access probe; do
    for rep in 1 2; do
      [ "$b" = access ] && [ "$rep" = 2 ] && continue
      opt=noprobes; [ "$b" = probe ] && opt=probes
      d=$(rundir "${b}_${p}_kc1_$rep")
      run "$d" "$d/run" "$WORK/b_${b}_$p/run_column" kc1 "$d" "$opt"
    done
  done
  # tables (access build)
  d=$(rundir "access_${p}_tables")
  run "$d" "$d/run" "$WORK/b_access_$p/run_dump_tables" "$d"
  # sweeps (probe build), twice
  for rep in 1 2; do
    d=$(rundir "probe_${p}_sweeps_$rep")
    for f in $SWEEPS; do
      run "$d" "$d/$f" "$WORK/b_probe_$p/run_sweeps" sweep "$f" "$d"
    done
  done
  # refusals, one process per case
  d=$(rundir "probe_${p}_refusals")
  for n in 1 2 3 4 5 6; do
    set +e
    ( cd "$d" && FSBM_REFUSAL_FILE="$d/refusal_$n.message" timeout "$RUN_TIMEOUT" \
        $NICE "$WORK/b_probe_$p/run_sweeps" refusal "$n" "$d" > "$d/refusal_$n.out" 2> "$d/refusal_$n.err" )
    echo $? > "$d/refusal_$n.status"
    set -e
  done
done
d=$(rundir probe_o64_sensitivity)
run "$d" "$d/run" "$WORK/b_probe_o64/run_column" sensitivity "$d"
d=$(rundir probe_o64_crossreplay)
run "$d" "$d/run" "$WORK/b_probe_o64/run_column" crossreplay "$d" "$WORK/r_probe_o32_kc1_1/kc1_handoff.f64"
# Hidden-state control: each build replayed on its OWN pre-call states in a
# fresh process must give back its own post-call records byte for byte, or
# FAST_SBM carries state between calls that the replay records do not hold.
for p in o64 o32; do
  d=$(rundir "probe_${p}_selfreplay")
  run "$d" "$d/run" "$WORK/b_probe_$p/run_column" crossreplay "$d" "$WORK/r_probe_${p}_kc1_1/kc1_handoff.f64"
done
if [ "$EVIDENCE" = 1 ]; then
  d=$(rundir wrfopt_o32_kc1_1)
  run "$d" "$d/run" "$WORK/b_wrfopt_o32/run_column" kc1 "$d" noprobes
fi

# 5b. Milestone 2: the cold sweeps and refusals, KC-2 and the graupel builds --
COLD_SWEEPS="freez melt ice_nucl jernucl01_ice jerrate_ice jertimesc_ice jerdfun_ice onecond2 onecond3 coll_cold modkrn coal_cold"
NREFC=13
for p in o64 o32; do
  # KC-2 in every hail build, twice for the pristine and probe builds
  for b in pristine access probe; do
    for rep in 1 2; do
      [ "$b" = access ] && [ "$rep" = 2 ] && continue
      opt=noprobes; [ "$b" = probe ] && opt=probes
      d=$(rundir "${b}_${p}_kc2_$rep")
      run "$d" "$d/run" "$WORK/b_${b}_$p/run_kc2" kc2 "$d" "$opt"
    done
  done
  # KC-2 in the graupel builds, twice each
  for b in gpristine gprobe; do
    for rep in 1 2; do
      opt=noprobes; [ "$b" = gprobe ] && opt=probes
      d=$(rundir "${b}_${p}_kc2g_$rep")
      run "$d" "$d/run" "$WORK/b_${b}_$p/run_kc2" kc2g "$d" "$opt"
    done
  done
  # the low-edge tile (i, j = 1..3): pristine once, probe twice
  for r in pristine_1 probe_1 probe_2; do
    b=${r%_*}
    opt=noprobes; [ "$b" = probe ] && opt=probes
    d=$(rundir "${b}_${p}_kc2edge_${r#*_}")
    run "$d" "$d/run" "$WORK/b_${b}_$p/run_kc2" kc2edge "$d" "$opt"
  done
  # cold sweeps (probe build) and cold COAL_BOTT_NEW in the graupel probe
  # build, twice each
  for rep in 1 2; do
    d=$(rundir "probe_${p}_coldsweeps_$rep")
    for f in $COLD_SWEEPS; do
      run "$d" "$d/$f" "$WORK/b_probe_$p/run_sweeps" sweep "$f" "$d"
    done
    d=$(rundir "gprobe_${p}_coldsweeps_$rep")
    run "$d" "$d/coal_cold" "$WORK/b_gprobe_$p/run_sweeps" sweep coal_cold "$d"
  done
  # cold refusals, one process per case
  d=$(rundir "probe_${p}_refusals_cold")
  for n in $(seq 1 $NREFC); do
    set +e
    ( cd "$d" && FSBM_REFUSAL_FILE="$d/refusal_cold_$n.message" timeout "$RUN_TIMEOUT" \
        $NICE "$WORK/b_probe_$p/run_sweeps" refusal_cold "$n" "$d" > "$d/refusal_cold_$n.out" \
        2> "$d/refusal_cold_$n.err" )
    echo $? > "$d/refusal_cold_$n.status"
    set -e
  done
done
d=$(rundir probe_o64_kc2_sensitivity)
run "$d" "$d/run" "$WORK/b_probe_o64/run_kc2" sensitivity "$d"
d=$(rundir probe_o64_kc2_crossreplay)
run "$d" "$d/run" "$WORK/b_probe_o64/run_kc2" crossreplay "$d" \
  "$WORK/r_probe_o32_kc2_1/kc2_handoff.f64" kc2
for p in o64 o32; do
  d=$(rundir "probe_${p}_kc2_selfreplay")
  run "$d" "$d/run" "$WORK/b_probe_$p/run_kc2" crossreplay "$d" \
    "$WORK/r_probe_${p}_kc2_1/kc2_handoff.f64" kc2
  d=$(rundir "gprobe_${p}_kc2g_selfreplay")
  run "$d" "$d/run" "$WORK/b_gprobe_$p/run_kc2" crossreplay "$d" \
    "$WORK/r_gprobe_${p}_kc2g_1/kc2g_handoff.f64" kc2g
done
if [ "$EVIDENCE" = 1 ]; then
  d=$(rundir wrfopt_o32_kc2_1)
  run "$d" "$d/run" "$WORK/b_wrfopt_o32/run_kc2" kc2 "$d" noprobes
fi

# 6. Controls -----------------------------------------------------------------
same() { cmp -s "$1" "$2" && echo true || echo false; }
{
  echo "{"
  first=1
  for p in o64 o32; do
    [ $first = 1 ] || echo ","
    first=0
    printf '  "determinism_pristine_%s_control": %s,\n' "$p" \
      "$(same "$WORK/r_pristine_${p}_kc1_1/kc1_control.raw" "$WORK/r_pristine_${p}_kc1_2/kc1_control.raw")"
    printf '  "inert_access_%s_control": %s,\n' "$p" \
      "$(same "$WORK/r_pristine_${p}_kc1_1/kc1_control.raw" "$WORK/r_access_${p}_kc1_1/kc1_control.raw")"
    printf '  "inert_probe_%s_control": %s,\n' "$p" \
      "$(same "$WORK/r_pristine_${p}_kc1_1/kc1_control.raw" "$WORK/r_probe_${p}_kc1_1/kc1_control.raw")"
    printf '  "inert_probe_run2_%s_control": %s' "$p" \
      "$(same "$WORK/r_pristine_${p}_kc1_1/kc1_control.raw" "$WORK/r_probe_${p}_kc1_2/kc1_control.raw")"
    for f in kc1_case kc1_summary kc1_replay kc1_control kc1_probe; do
      echo ","
      printf '  "determinism_probe_%s_%s_raw": %s' "$p" "$f" \
        "$(same "$WORK/r_probe_${p}_kc1_1/$f.raw" "$WORK/r_probe_${p}_kc1_2/$f.raw")"
      echo ","
      printf '  "determinism_probe_%s_%s_layout": %s' "$p" "$f" \
        "$(same "$WORK/r_probe_${p}_kc1_1/$f.layout" "$WORK/r_probe_${p}_kc1_2/$f.layout")"
    done
    echo ","
    printf '  "determinism_probe_%s_handoff": %s' "$p" \
      "$(same "$WORK/r_probe_${p}_kc1_1/kc1_handoff.f64" "$WORK/r_probe_${p}_kc1_2/kc1_handoff.f64")"
    for f in $SWEEPS; do
      echo ","
      printf '  "determinism_sweep_%s_%s": %s' "$p" "$f" \
        "$(same "$WORK/r_probe_${p}_sweeps_1/sweep_$f.raw" "$WORK/r_probe_${p}_sweeps_2/sweep_$f.raw")"
    done
  done
  if [ "$EVIDENCE" = 1 ]; then
    echo ","
    printf '  "wrfopt_o32_control_identical_to_o32": %s' \
      "$(same "$WORK/r_pristine_o32_kc1_1/kc1_control.raw" "$WORK/r_wrfopt_o32_kc1_1/kc1_control.raw")"
  fi
  # milestone 2
  for p in o64 o32; do
    r="$WORK/r"
    echo ","
    printf '  "determinism_pristine_%s_kc2_control": %s,\n' "$p" \
      "$(same "${r}_pristine_${p}_kc2_1/kc2_control.raw" "${r}_pristine_${p}_kc2_2/kc2_control.raw")"
    printf '  "inert_access_%s_kc2_control": %s,\n' "$p" \
      "$(same "${r}_pristine_${p}_kc2_1/kc2_control.raw" "${r}_access_${p}_kc2_1/kc2_control.raw")"
    printf '  "inert_probe_%s_kc2_control": %s,\n' "$p" \
      "$(same "${r}_pristine_${p}_kc2_1/kc2_control.raw" "${r}_probe_${p}_kc2_1/kc2_control.raw")"
    printf '  "inert_probe_run2_%s_kc2_control": %s,\n' "$p" \
      "$(same "${r}_pristine_${p}_kc2_1/kc2_control.raw" "${r}_probe_${p}_kc2_2/kc2_control.raw")"
    printf '  "determinism_gpristine_%s_kc2g_control": %s,\n' "$p" \
      "$(same "${r}_gpristine_${p}_kc2g_1/kc2g_control.raw" "${r}_gpristine_${p}_kc2g_2/kc2g_control.raw")"
    printf '  "inert_gprobe_%s_kc2g_control": %s,\n' "$p" \
      "$(same "${r}_gpristine_${p}_kc2g_1/kc2g_control.raw" "${r}_gprobe_${p}_kc2g_1/kc2g_control.raw")"
    printf '  "inert_gprobe_run2_%s_kc2g_control": %s,\n' "$p" \
      "$(same "${r}_gpristine_${p}_kc2g_1/kc2g_control.raw" "${r}_gprobe_${p}_kc2g_2/kc2g_control.raw")"
    printf '  "graupel_build_differs_from_hail_%s": %s' "$p" \
      "$(cmp -s "${r}_pristine_${p}_kc2_1/kc2_control.raw" "${r}_gpristine_${p}_kc2g_1/kc2g_control.raw" \
         && echo false || echo true)"
    for f in kc2_case kc2_summary kc2_replay kc2_control kc2_probe; do
      echo ","
      printf '  "determinism_probe_%s_%s_raw": %s,\n' "$p" "$f" \
        "$(same "${r}_probe_${p}_kc2_1/$f.raw" "${r}_probe_${p}_kc2_2/$f.raw")"
      printf '  "determinism_probe_%s_%s_layout": %s' "$p" "$f" \
        "$(same "${r}_probe_${p}_kc2_1/$f.layout" "${r}_probe_${p}_kc2_2/$f.layout")"
    done
    echo ","
    printf '  "determinism_probe_%s_kc2_handoff": %s' "$p" \
      "$(same "${r}_probe_${p}_kc2_1/kc2_handoff.f64" "${r}_probe_${p}_kc2_2/kc2_handoff.f64")"
    for f in kc2g_case kc2g_summary kc2g_replay kc2g_control kc2g_probe; do
      echo ","
      printf '  "determinism_gprobe_%s_%s_raw": %s' "$p" "$f" \
        "$(same "${r}_gprobe_${p}_kc2g_1/$f.raw" "${r}_gprobe_${p}_kc2g_2/$f.raw")"
    done
    for f in $COLD_SWEEPS; do
      echo ","
      printf '  "determinism_sweep_%s_%s": %s' "$p" "$f" \
        "$(same "${r}_probe_${p}_coldsweeps_1/sweep_$f.raw" "${r}_probe_${p}_coldsweeps_2/sweep_$f.raw")"
    done
    echo ","
    printf '  "determinism_sweep_%s_coal_cold_graupel": %s' "$p" \
      "$(same "${r}_gprobe_${p}_coldsweeps_1/sweep_coal_cold.raw" "${r}_gprobe_${p}_coldsweeps_2/sweep_coal_cold.raw")"
    echo ","
    printf '  "inert_probe_%s_kc2edge_control": %s,\n' "$p" \
      "$(same "${r}_pristine_${p}_kc2edge_1/kc2edge_control.raw" "${r}_probe_${p}_kc2edge_1/kc2edge_control.raw")"
    printf '  "inert_probe_run2_%s_kc2edge_control": %s' "$p" \
      "$(same "${r}_pristine_${p}_kc2edge_1/kc2edge_control.raw" "${r}_probe_${p}_kc2edge_2/kc2edge_control.raw")"
    for f in kc2edge_replay kc2edge_probe; do
      echo ","
      printf '  "determinism_probe_%s_%s_raw": %s,\n' "$p" "$f" \
        "$(same "${r}_probe_${p}_kc2edge_1/$f.raw" "${r}_probe_${p}_kc2edge_2/$f.raw")"
      printf '  "determinism_probe_%s_%s_layout": %s' "$p" "$f" \
        "$(same "${r}_probe_${p}_kc2edge_1/$f.layout" "${r}_probe_${p}_kc2edge_2/$f.layout")"
    done
  done
  if [ "$EVIDENCE" = 1 ]; then
    echo ","
    printf '  "wrfopt_o32_kc2_control_identical_to_o32": %s' \
      "$(same "$WORK/r_pristine_o32_kc2_1/kc2_control.raw" "$WORK/r_wrfopt_o32_kc2_1/kc2_control.raw")"
  fi
  echo
  echo "}"
} > "$OUT/controls.json"
for p in o64 o32; do
  for s in pristine_${p}_kc2_1 pristine_${p}_kc2_2 access_${p}_kc2_1 probe_${p}_kc2_1 probe_${p}_kc2_2; do
    printf '%s  %s\n' "$(sha256sum "$WORK/r_$s/kc2_control.raw" | cut -c1-64)" "$s/kc2_control.raw"
  done
  for s in gpristine_${p}_kc2g_1 gpristine_${p}_kc2g_2 gprobe_${p}_kc2g_1 gprobe_${p}_kc2g_2; do
    printf '%s  %s\n' "$(sha256sum "$WORK/r_$s/kc2g_control.raw" | cut -c1-64)" "$s/kc2g_control.raw"
  done
  for s in pristine_${p}_kc2edge_1 probe_${p}_kc2edge_1 probe_${p}_kc2edge_2; do
    printf '%s  %s\n' "$(sha256sum "$WORK/r_$s/kc2edge_control.raw" | cut -c1-64)" "$s/kc2edge_control.raw"
  done
done > "$OUT/control_sha256_kc2.txt"
[ "$EVIDENCE" = 1 ] && printf '%s  %s\n' "$(sha256sum "$WORK/r_wrfopt_o32_kc2_1/kc2_control.raw" | cut -c1-64)" \
  "wrfopt_o32_kc2_1/kc2_control.raw" >> "$OUT/control_sha256_kc2.txt"
for p in o64 o32; do
  for s in pristine_${p}_kc1_1 pristine_${p}_kc1_2 access_${p}_kc1_1 probe_${p}_kc1_1 probe_${p}_kc1_2; do
    printf '%s  %s\n' "$(sha256sum "$WORK/r_$s/kc1_control.raw" | cut -c1-64)" "$s/kc1_control.raw"
  done
done > "$OUT/control_sha256.txt"
[ "$EVIDENCE" = 1 ] && printf '%s  %s\n' "$(sha256sum "$WORK/r_wrfopt_o32_kc1_1/kc1_control.raw" | cut -c1-64)" \
  "wrfopt_o32_kc1_1/kc1_control.raw" >> "$OUT/control_sha256.txt"

# 7. Collect the streams pack_fixtures.py reads --------------------------------
for p in o64 o32; do
  mkdir -p "$OUT/$p"
  cp "$WORK/r_access_${p}_tables"/{tables,kernels}.{raw,layout} "$OUT/$p/"
  for f in kc1_case kc1_summary kc1_replay kc1_probe; do
    cp "$WORK/r_probe_${p}_kc1_1/$f".{raw,layout} "$OUT/$p/"
  done
  for f in $SWEEPS; do
    cp "$WORK/r_probe_${p}_sweeps_1/sweep_$f".{raw,layout} "$OUT/$p/"
  done
  mkdir -p "$OUT/$p/refusals"
  for n in 1 2 3 4 5 6; do
    r="$WORK/r_probe_${p}_refusals"
    cp "$r/refusal_$n".{raw,layout,status,out,err} "$OUT/$p/refusals/" 2>/dev/null || true
    [ -f "$r/refusal_$n.message" ] && cp "$r/refusal_$n.message" "$OUT/$p/refusals/" || true
  done
done
cp "$WORK/r_probe_o64_sensitivity/kc1_sensitivity".{raw,layout} "$OUT/o64/"
cp "$WORK/r_probe_o64_crossreplay/kc1_crossreplay".{raw,layout} "$OUT/o64/"
for p in o64 o32; do
  cp "$WORK/r_probe_${p}_selfreplay/kc1_crossreplay.raw" "$OUT/$p/kc1_selfreplay.raw"
  cp "$WORK/r_probe_${p}_selfreplay/kc1_crossreplay.layout" "$OUT/$p/kc1_selfreplay.layout"
done
if [ "$EVIDENCE" = 1 ]; then
  mkdir -p "$OUT/o32wrf"
  cp "$WORK/r_wrfopt_o32_kc1_1/kc1_summary".{raw,layout} "$OUT/o32wrf/"
  cp "$WORK/r_pristine_o32_kc1_1/kc1_summary.raw" "$OUT/o32wrf/pristine_kc1_summary.raw"
  cp "$WORK/r_pristine_o32_kc1_1/kc1_summary.layout" "$OUT/o32wrf/pristine_kc1_summary.layout"
fi
# milestone 2 streams
for p in o64 o32; do
  for f in kc2_case kc2_summary kc2_replay kc2_probe; do
    cp "$WORK/r_probe_${p}_kc2_1/$f".{raw,layout} "$OUT/$p/"
  done
  for f in kc2g_case kc2g_summary kc2g_replay kc2g_probe; do
    cp "$WORK/r_gprobe_${p}_kc2g_1/$f".{raw,layout} "$OUT/$p/"
  done
  for f in kc2edge_replay kc2edge_probe; do
    cp "$WORK/r_probe_${p}_kc2edge_1/$f".{raw,layout} "$OUT/$p/"
  done
  for f in $COLD_SWEEPS; do
    cp "$WORK/r_probe_${p}_coldsweeps_1/sweep_$f".{raw,layout} "$OUT/$p/"
  done
  cp "$WORK/r_gprobe_${p}_coldsweeps_1/sweep_coal_cold.raw" "$OUT/$p/sweep_coal_cold_graupel.raw"
  cp "$WORK/r_gprobe_${p}_coldsweeps_1/sweep_coal_cold.layout" "$OUT/$p/sweep_coal_cold_graupel.layout"
  mkdir -p "$OUT/$p/refusals_cold"
  for n in $(seq 1 $NREFC); do
    r="$WORK/r_probe_${p}_refusals_cold"
    cp "$r/refusal_cold_$n".{raw,layout,status,out,err} "$OUT/$p/refusals_cold/" 2>/dev/null || true
    [ -f "$r/refusal_cold_$n.message" ] && cp "$r/refusal_cold_$n.message" "$OUT/$p/refusals_cold/" || true
  done
  cp "$WORK/r_probe_${p}_kc2_selfreplay/kc2_crossreplay.raw" "$OUT/$p/kc2_selfreplay.raw"
  cp "$WORK/r_probe_${p}_kc2_selfreplay/kc2_crossreplay.layout" "$OUT/$p/kc2_selfreplay.layout"
  cp "$WORK/r_gprobe_${p}_kc2g_selfreplay/kc2g_crossreplay.raw" "$OUT/$p/kc2g_selfreplay.raw"
  cp "$WORK/r_gprobe_${p}_kc2g_selfreplay/kc2g_crossreplay.layout" "$OUT/$p/kc2g_selfreplay.layout"
done
cp "$WORK/r_probe_o64_kc2_sensitivity/kc2_sensitivity".{raw,layout} "$OUT/o64/"
cp "$WORK/r_probe_o64_kc2_crossreplay/kc2_crossreplay".{raw,layout} "$OUT/o64/"
if [ "$EVIDENCE" = 1 ]; then
  cp "$WORK/r_wrfopt_o32_kc2_1/kc2_summary".{raw,layout} "$OUT/o32wrf/"
  cp "$WORK/r_pristine_o32_kc2_1/kc2_summary.raw" "$OUT/o32wrf/pristine_kc2_summary.raw"
  cp "$WORK/r_pristine_o32_kc2_1/kc2_summary.layout" "$OUT/o32wrf/pristine_kc2_summary.layout"
fi
cp "$SRC"/module_mp_fast_sbm_access.F "$SRC"/module_mp_fast_sbm_probe.F "$OUT/" 2>/dev/null || true
cp "$SRC"/module_mp_fast_sbm_graupel.F "$SRC"/module_mp_fast_sbm_graupel_probe.F "$OUT/" 2>/dev/null || true

# 8. Build information ----------------------------------------------------------
# The harness files that produced these streams; pack_fixtures.py refuses to
# pack unless the files beside it hash the same.
( cd "$HERE" && sha256sum build.sh sources.sha256 stubs_wrf.F90 transform_access.py \
    instrument_probes.py fsbm_probe.F90 run_dump_tables.F90 run_sweeps.F90 \
    run_column.F90 transform_graupel.py run_kc2.F90 ) > "$OUT/harness.sha256"
T1=$(date +%s)
python3 - "$OUT/build_info.json" <<EOF
import json, platform, subprocess, sys
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
info = {
    "compiler": sh("gfortran --version | head -1"),
    "host": platform.node(),
    "kernel": platform.platform(),
    "cpu": sh("lscpu | awk -F: '/Model name/ {gsub(/^ +/, \"\", \$2); print \$2; exit}'"),
    "date_utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
    "wall_seconds": $T1 - $T0,
    "flags": {
        "common": "$COMMON",
        "o64": "$O64",
        "o32": "$O32",
        "o32_wrf": "$OWRF",
        "fsbm_io": "$IOFLAGS (fsbm_io is always compiled without -freal-4-real-8)",
    },
    "fast_sbm_compile_warnings": {
        b: int(open("$WORK/b_%s/warnings.count" % b).read().strip() or 0)
        for b in ["pristine_o64", "pristine_o32", "access_o64", "access_o32",
                  "probe_o64", "probe_o32", "gpristine_o64", "gpristine_o32",
                  "gprobe_o64", "gprobe_o32"]
    },
    "evidence_build": bool($EVIDENCE),
    "run_timeout_seconds": $RUN_TIMEOUT,
}
json.dump(info, open(sys.argv[1], "w"), indent=2, sort_keys=True)
EOF
echo "build.sh: done in $((T1 - T0)) s; output in $OUT"
