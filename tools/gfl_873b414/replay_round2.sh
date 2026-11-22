#!/bin/bash
# Round 2: strict proof mode against the pristine glibc 2.39 referee, default (fast) mode against the
# correctly rounded witness, and the kernel timing beside today's cu_physics=3 kernel.
L=${GFL_WORK:?set GFL_WORK to the folder holding captures/, replays/ and venv/}; W=${GFL_TREE:-$(pwd)}
V=${CUDA_VENV:?set CUDA_VENV to a venv with the nvidia cu13 wheels}
export LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib
cd $W; mkdir -p $L/reports2
for s in 000001_000001 000002_000002 000003_000003; do
  CUPY_CACHE_DIR=$L/cupy-strict GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 $L/venv/bin/python tools/gfl_873b414/replay_gfl.py $L/captures/pre_$s.bin $L/captures/post_$s.bin $L/reports2/strict-vs-native-glibc239-$s.json
  CUPY_CACHE_DIR=$L/cupy-fast GPUWM_NUMERICS=fast GPUWM_WRF_EXACT=0 $L/venv/bin/python tools/gfl_873b414/replay_gfl.py $L/captures/pre_$s.bin $L/replays/native_crgamma_$s.bin $L/reports2/fast-vs-native-crgamma-$s.json
done
CUPY_CACHE_DIR=$L/cupy-fast GPUWM_NUMERICS=fast GPUWM_WRF_EXACT=0 $L/venv/bin/python tools/gfl_873b414/time_gf_vs_gfl.py $L/captures/pre_000003_000003.bin $L/reports2/timing-fast-step3.json 9
