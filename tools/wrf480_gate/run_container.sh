#!/bin/bash
# usage: EXTRA_ENV="-e K=V" run_container.sh CASE OUTNAME [door args...]
# ArWen run inside the glibc 2.39 referee container (localhost/wrfref2404:1),
# the libm platform the native WRF 4.8.0 strict referee runs on.
C=$1; OUT=$2; shift 2
P=${WRF480_ROOT:?set WRF480_ROOT to the referee working root}
E=${WOOF_TREE:?set WOOF_TREE to the engine checkout}
PY=${GATE_PYTHON:?set GATE_PYTHON to the venv python that runs the engine}
CU=${GATE_CUDA_PATH:?set GATE_CUDA_PATH to the nvidia cu13 wheel root}
G=$P/gate
T=$E/tools/rustwx/target-c239/release
rm -rf $G/$OUT; mkdir -p $G/tmp
podman run --rm --device nvidia.com/gpu=all --userns=keep-id -v "$HOME:$HOME" -w "$E" \
 -e HOME="$HOME" -e TMPDIR=$G/tmp -e PYTHONPATH="$E" \
 -e CUDA_PATH="$CU" \
 -e CUPY_CACHE_DIR=$P/captures/.cupy -e GPUWM_HISTORY_COMPRESSION=none \
 -e WOOF_STATIC_BRIDGE=$T/libstatic_fields.so -e WOOF_NCWRITE_BRIDGE=$T/libnetcdf_writer.so \
 $EXTRA_ENV localhost/wrfref2404:1 bash -c "ldd --version | head -1; $PY -m woof.wrfinput_forecast --wrfinput $G/$C-in --outdir $G/$OUT --products none $*; echo EXIT \$?"
