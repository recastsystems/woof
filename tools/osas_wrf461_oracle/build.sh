#!/usr/bin/env bash
# Build the WRF v4.6.1 CU_OSAS (cu_physics = 95) column oracle.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# The four Fortran files are byte-pinned against the v4.6.1 tag (d66e442).
# They go through WRF's own preprocessing pipeline (the sed comment filter,
# /lib/cpp -P -nostdinc -traditional-cpp with the em_real defines of a
# serial configure-32 build, tools/standard.exe, cpp again), and the
# resulting .f90 files are pinned too: they must equal, byte for byte, what
# the stock WRF build compiles (phys/*.f90 of a configure-32 em_real build).
#
# Compiled with the WRF 0 ULP referee's arithmetic flags (configure.wrf
# FCOPTIM of the scalar build): -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize, plus FORMAT_FREE and FCCOMPAT.  BYTESWAPIO
# (-fconvert=big-endian) is left out: it changes only how unformatted files
# are written, and the oracle's dumps are little-endian streams.
#
# The harness edits (run_osas.F90's header lists them) are three sed
# expressions applied to the preprocessed text, each checked to have fired
# exactly the expected number of times.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

declare -A pinned_src=(
  [module_gfs_machine]="300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25"
  [module_gfs_physcons]="2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52"
  [module_gfs_funcphys]="0d43dccf0510bd49ef07f81c357e4b96fd9391604f7c14d7e3cf8700ff383e74"
  [module_cu_osas]="f8cc3fbd6277cf886bcdffa9e8df7623884dfb1886acdbfd0d2471329c3cfd50"
)
declare -A pinned_f90=(
  [module_gfs_machine]="f82c87d1826d8b9c82e304df75e790836946269eb36637c19380c2b64b8ebc2d"
  [module_gfs_physcons]="25800d925187b65ba784d9641a1884f0f66348fd6542e183ce3f6876175756c3"
  [module_gfs_funcphys]="558e901adea6e734d15caafebbd2b1386e5419e7244fea88b128a41e05e9c9d6"
  [module_cu_osas]="0df3c55f33cf6d1a920c43cc40165dc9c76d3244730061941d939d77384b6a3c"
)
order=(module_gfs_machine module_gfs_physcons module_gfs_funcphys module_cu_osas)

for name in "${order[@]}"; do
    src="${source_root}/phys/${name}.F"
    actual=$(sha256sum "${src}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned_src[$name]}" ]]; then
        echo "phys/${name}.F is ${actual}, not the pinned ${pinned_src[$name]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"

# WRF's preprocessing, as phys/Makefile runs it for a serial configure-32
# em_real build (the defines are copied from that build's compile log).
gcc -O2 -o standard.exe "${source_root}/tools/standard.c"
defs="-DEM_CORE=1 -DNMM_CORE=0 -DNMM_MAX_DIM=2600 -DDA_CORE=0 -DWRFPLUS=0 \
-DIWORDSIZE=4 -DDWORDSIZE=8 -DRWORDSIZE=4 -DLWORDSIZE=4 -DNONSTANDARD_SYSTEM_SUBR \
-DWRF_USE_CLM -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT -DNETCDF \
-DLANDREAD_STUB=1 -DUSE_ALLOCATABLES -Dwrfmodel -DGRIB1 -DINTIO -DKEEP_INT_AROUND \
-DLIMIT_ARGS -DBUILD_RRTMG_FAST=0 -DBUILD_RRTMK=0 -DBUILD_SBM_FAST=1 \
-DSHOW_ALL_VARS_USED=0 -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25 \
-DNMM_NEST=0"
mkdir -p inc
for name in "${order[@]}"; do
    sed -e "s/^\!.*'.*//" -e "s/^ *\!.*'.*//" "${source_root}/phys/${name}.F" > "${name}.G"
    # shellcheck disable=SC2086
    /lib/cpp -P -nostdinc -I"${build_dir}/inc" ${defs} -I. -traditional-cpp \
        -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT "${name}.G" > "${name}.bb"
    ./standard.exe "${name}.bb" | /lib/cpp -P -nostdinc -traditional-cpp \
        -DUSE_NETCDF4_FEATURES -DWRFIO_NCD_LARGE_FILE_SUPPORT > "${name}.f90"
    actual=$(sha256sum "${name}.f90" | awk '{print $1}')
    if [[ "${actual}" != "${pinned_f90[$name]}" ]]; then
        echo "${name}.f90 is ${actual}, not the stock build's ${pinned_f90[$name]}" >&2
        exit 4
    fi
done

# --- harness edits ----------------------------------------------------------
fired() {  # file pattern expected
    local n
    n=$(grep -c -- "$2" "$1" || true)
    if [[ "${n}" != "$3" ]]; then
        echo "harness edit '$2' fired ${n} times in $1, expected $3" >&2
        exit 5
    fi
}
cp module_cu_osas.f90 module_cu_osas_oracle.f90
sed -i -e 's/^\( *\)call init_random_seed()$/\1call osas_oracle_seed()/' \
       -e 's/^\( *\)call random_number(XKT2)$/\1call osas_oracle_xkt2(XKT2)/' \
       module_cu_osas_oracle.f90
sed -i -e '/^ *CALL SHALCV(IM,IM,KX,DELT,DEL,PRSI,PRSL,PRSLK,KUO,Q1,T1,DPSHC)/{
i\      call osas_oracle_after_deep(IM,KX,T1,Q1,QL,RN,KBOT,KTOP,KUO,CLDWRK)
a\      call osas_oracle_after_shallow(IM,KX,T1,Q1)
}' module_cu_osas_oracle.f90
fired module_cu_osas_oracle.f90 'call osas_oracle_seed()' 1
fired module_cu_osas_oracle.f90 'call osas_oracle_xkt2(XKT2)' 1
fired module_cu_osas_oracle.f90 'random_number' 0
fired module_cu_osas_oracle.f90 'call osas_oracle_after_deep' 1
fired module_cu_osas_oracle.f90 'call osas_oracle_after_shallow' 1

cp module_gfs_funcphys.f90 module_gfs_funcphys_oracle.f90
sed -i -e 's/^\( *\)public gfuncphys$/\1public gfuncphys, osas_oracle_dump_tables/' \
    module_gfs_funcphys_oracle.f90
last=$(grep -n -i '^ *end module' module_gfs_funcphys_oracle.f90 | tail -1 | cut -d: -f1)
head -n $((last - 1)) module_gfs_funcphys_oracle.f90 > tmp.f90
cat >> tmp.f90 <<'EOF'
  subroutine osas_oracle_dump_tables(path)
    implicit none
    character(len=*), intent(in) :: path
    integer :: u
    open(newunit=u, file=path, access='stream', form='unformatted', status='replace')
    write(u) c1xpvs, c2xpvs, c1xtdp, c2xtdp, c1xthe, c2xthe, c1ythe, c2ythe
    write(u) c1xma, c2xma, c1yma, c2yma, c1xtlcl, c2xtlcl, c1ytlcl, c2ytlcl
    write(u) tbpvs, tbtdp, tbthe, tbtma, tbqma, tbtlcl
    close(u)
  end subroutine osas_oracle_dump_tables
EOF
tail -n +"${last}" module_gfs_funcphys_oracle.f90 >> tmp.f90
mv tmp.f90 module_gfs_funcphys_oracle.f90
fired module_gfs_funcphys_oracle.f90 'public gfuncphys, osas_oracle_dump_tables' 1

# --- compile: the WRF 0 ULP referee's arithmetic flags -----------------------
FC_ARITH="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"
FC_FORM="-w -ffree-form -ffree-line-length-none -fallow-argument-mismatch -fallow-invalid-boz"
# shellcheck disable=SC2086
{
gfortran -c ${FC_ARITH} ${FC_FORM} module_gfs_machine.f90
gfortran -c ${FC_ARITH} ${FC_FORM} module_gfs_physcons.f90
gfortran -c ${FC_ARITH} ${FC_FORM} -o module_gfs_funcphys.o module_gfs_funcphys_oracle.f90
gfortran -c ${FC_ARITH} ${FC_FORM} -o module_cu_osas.o module_cu_osas_oracle.f90
gfortran -c ${FC_ARITH} ${FC_FORM} "${script_dir}/run_osas.F90"
gfortran -o run_osas run_osas.o module_cu_osas.o module_gfs_funcphys.o \
    module_gfs_physcons.o module_gfs_machine.o
}

# Receipts: the toolchain, and the libm entry points the scheme reaches.
{
    gfortran --version | sed -n '1p'
    ldd --version | sed -n '1p'
    grep -m1 'model name' /proc/cpuinfo || true
} > toolchain.txt
nm -u module_cu_osas.o module_gfs_funcphys.o | sort -u > undefined.txt
if grep -q '_ZGV' undefined.txt; then
    echo "a libmvec vector symbol reached the oracle objects" >&2
    exit 6
fi
echo "built ${build_dir}/run_osas"
