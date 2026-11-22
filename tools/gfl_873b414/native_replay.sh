#!/bin/bash
# Inside wrfref2404:1: replay each captured PRE through the referee objects (glibc and CR-gamma witness).
ulimit -s unlimited; cd /l
for pre in captures/pre_*.bin; do
  s=$(basename $pre .bin | sed 's/^pre_//')
  ./native-replay/gfl_replay_native_glibc $pre replays/native_glibc_$s.bin
  ./native-replay/gfl_replay_native_crgamma $pre replays/native_crgamma_$s.bin
done
ls -la replays
