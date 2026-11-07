#!/bin/bash
set -euo pipefail
cd /workspace/runs/port-dyn-options
. /workspace/env.sh
. venv/bin/activate
mkdir -p hsca2-init-witness
cp /workspace/native-wrf471/dyn_em/module_advect_em.F hsca2-init-witness/module_advect_em.F
cp /workspace/native-wrf471/share/module_model_constants.F hsca2-init-witness/module_model_constants.F
cp /workspace/native-wrf471/frame/module_wrf_error.F hsca2-init-witness/module_wrf_error.F
cp engine/tools/advect_wrf471_oracle/{stub_wrf.F90,run_advect.F90} hsca2-init-witness/
cd hsca2-init-witness
flags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4)
for init in zero snan; do
 mkdir -p "$init"
 (
 cd "$init"
 gfortran "${flags[@]}" -finit-real="$init" -c ../stub_wrf.F90
 gfortran "${flags[@]}" -finit-real="$init" -c ../module_model_constants.F
 gfortran "${flags[@]}" -finit-real="$init" -c ../module_wrf_error.F
 gfortran "${flags[@]}" -finit-real="$init" -c ../module_advect_em.F
 gfortran "${flags[@]}" -finit-real="$init" -c ../run_advect.F90
 gfortran -o run_advect *.o
 printf '%s\n' "${flags[*]} -finit-real=$init" > compiler-flags.txt
 gfortran --version > compiler-version.txt
 sha256sum ../*.F ../*.F90 *.o run_advect > hashes.txt
 )
done
python ../hsca2_init_witness.py
