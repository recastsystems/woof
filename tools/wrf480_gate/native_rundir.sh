#!/bin/bash
# usage: native_rundir.sh CASE RUNDIR [WRF_BUILD]
# Builds the native WRF 4.8.0 referee run dir for a gate case: the build's run/
# tables, main/wrf.exe, the case's wrfinput/wrfbdy (gate/<CASE>-in, our own init),
# the case deck cases/<CASE>/namelist.input and, when the deck names an iofields
# file, that file from cases/<CASE>/. Without the iofields file native writes no
# PBL/cumulus/radiation tendencies or NOAHRES, so the gate compares a short history
# (A071: 195 fields, 344,854,371 words instead of 215 fields, 654,871,971 words);
# a deck that names one and has none here is refused.
# Then: run_native.sh RUNDIR NAME (pinned container localhost/wrfref2404:1).
set -e
C=$1; R=$2
P=${WRF480_ROOT:?set WRF480_ROOT to the referee working root}
B=${3:-$P/build/WRF-4.8.0-serial-pristine}
G=$P/gate
H=$(cd "$(dirname "$0")" && pwd)/cases/$C
[ -n "$C" ] && [ -n "$R" ] || { echo "usage: native_rundir.sh CASE RUNDIR [WRF_BUILD]" >&2; exit 2; }
[ -f "$H/namelist.input" ] || { echo "native_rundir: no deck $H/namelist.input" >&2; exit 2; }
io=$(sed -n "s/^ *iofields_filename *= *'\([^']*\)'.*/\1/p" "$H/namelist.input" | head -1)
if [ -n "$io" ] && [ ! -f "$H/$io" ]; then
  echo "native_rundir: deck names iofields '$io' but $H/$io is missing; native would write a short history" >&2; exit 2
fi
rm -rf "$R"; mkdir -p "$R"
for f in "$B"/run/*; do
  n=$(basename "$f")
  case $n in wrf.exe|namelist.input|wrfinput_d0*|wrfbdy_d0*|wrfout_*|rsl.*) continue;; esac
  ln -s "$f" "$R/$n"
done
cp "$B/main/wrf.exe" "$G/$C-in/wrfinput_d01" "$G/$C-in/wrfbdy_d01" "$R/"
cp "$H/namelist.input" "$R/namelist.input"
[ -z "$io" ] || cp "$H/$io" "$R/$io"
cd "$R"; sha256sum wrf.exe wrfinput_d01 wrfbdy_d01 namelist.input ${io:+"$io"} | tee SHA256-inputs.txt
