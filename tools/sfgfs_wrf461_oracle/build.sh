#!/usr/bin/env bash
# Build and run the GFS surface-layer oracle (sf_sfclay_physics=3) against
# the byte-unmodified WRF v4.6.1 phys/module_sf_gfs.F.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]
#
# The scheme is compiled the way WRF's own scalar referee build compiles it:
# the same sed/cpp/standard.exe preprocessing pipeline (the .f90 it produces
# is checked byte-equal to the referee's when WRF_BUILD_ROOT is given) and
# the same FCOPTIM, -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize.  Fortran REAL is 4 bytes (no -fdefault-real-8), as in
# WRF, so every default-kind literal in the GFS modules is a float32 value
# widened into the kind_phys (binary64) column.
#
# When WRF_BUILD_ROOT is given, a second driver is linked against that
# build's OWN object files (phys/module_sf_gfs.o, module_gfs_*.o,
# module_progtm.o, share/module_model_constants.o) and its outputs must be
# byte-identical to the from-source driver's: the oracle is then the
# artefact wrf.exe runs, not a reconstruction of it.
#
# The byte pins are the FILES of the pristine v4.6.1 tree (d66e442).
# module_surface_driver.F is pinned because gfs_seaice_wrapper.F90 carries
# two of its subroutines' statements (see that file's header).
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build_root=${3:+$(realpath "$3")}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

declare -A pins=(
    [tools/standard.c]=a766c03a8049f906b848d6ae10c9ef2d8d57944f1e87396a14de90fcfa306f37
    [phys/module_sf_gfs.F]=10eb3f3b47deb8f33d10e13a5b274db89a59cf5fb8c86a92f353205b9e73440e
    [phys/module_gfs_funcphys.F]=0d43dccf0510bd49ef07f81c357e4b96fd9391604f7c14d7e3cf8700ff383e74
    [phys/module_gfs_physcons.F]=2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52
    [phys/module_gfs_machine.F]=300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25
    [phys/module_progtm.F]=6d6c411638011f92f21764418bc7993ac7eefe8f05cfe79e0df05a5ab58d1b47
    [share/module_model_constants.F]=5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062
    [phys/module_surface_driver.F]=fde0533233b2b11524ef78da1e48178c867bde0de08a4de03d97d410b103b4fb
)
for rel in "${!pins[@]}"; do
    actual=$(sha256sum "${source_root}/${rel}" | awk '{print $1}')
    if [[ "${actual}" != "${pins[$rel]}" ]]; then
        echo "${rel} is ${actual}, not the pinned v4.6.1 ${pins[$rel]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- WRF's preprocessing, exactly (phys/Makefile via configure.wrf) ---------
gcc -O2 -o standard.exe "${source_root}/tools/standard.c"
cppflags=(-P -nostdinc -traditional-cpp -DEM_CORE=1 -DNMM_CORE=0
          -DNMM_MAX_DIM=2600 -DDA_CORE=0 -DWRFPLUS=0 -DIWORDSIZE=4
          -DDWORDSIZE=8 -DRWORDSIZE=4 -DLWORDSIZE=4)
fcflags=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w
         -ffree-form -ffree-line-length-none -fconvert=big-endian
         -frecord-marker=4)
preprocess() {   # $1 = path relative to the source root, writes <stem>.f90
    local rel=$1 stem
    stem=$(basename "${rel}" .F)
    sed -e "s/^\!.*'.*//" -e "s/^ *\!.*'.*//" "${source_root}/${rel}" > "${stem}.G"
    cpp "${cppflags[@]}" "${stem}.G" > "${stem}.bb"
    ./standard.exe "${stem}.bb" | cpp -P -nostdinc -traditional-cpp > "${stem}.f90"
    rm -f "${stem}.G" "${stem}.bb"
}
units=(share/module_model_constants.F phys/module_gfs_machine.F
       phys/module_gfs_physcons.F phys/module_gfs_funcphys.F
       phys/module_progtm.F phys/module_sf_gfs.F)
for rel in "${units[@]}"; do
    preprocess "${rel}"
    stem=$(basename "${rel}" .F)
    gfortran -c "${fcflags[@]}" -o "${stem}.o" "${stem}.f90"
    if [[ -n "${wrf_build_root}" ]]; then
        cmp "${stem}.f90" "${wrf_build_root}/${rel%.F}.f90"
    fi
done

gfortran -c "${fcflags[@]}" -I . -o gfs_seaice_wrapper.o \
    "${script_dir}/gfs_seaice_wrapper.F90"
gfortran -c "${fcflags[@]}" -I . -o run_sf_gfs.o "${script_dir}/run_sf_gfs.F90"
gcc -c -O2 -o read_tbpvs.o "${script_dir}/read_tbpvs.c"
objs=(module_model_constants.o module_gfs_machine.o module_gfs_physcons.o
      module_gfs_funcphys.o module_progtm.o module_sf_gfs.o)
gfortran -o run_sf_gfs run_sf_gfs.o gfs_seaice_wrapper.o read_tbpvs.o "${objs[@]}"

# --- receipts ------------------------------------------------------------------
nm -u module_sf_gfs.o | sed 's/^ *//' | sort > undefined-sf_gfs.txt
nm -u module_gfs_funcphys.o | sed 's/^ *//' | sort > undefined-funcphys.txt
if grep -q '_ZGV' undefined-sf_gfs.txt undefined-funcphys.txt; then
    echo "a libmvec vector symbol reached the scalar build" >&2
    exit 4
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# libm entry points module_sf_gfs.o calls:"
    grep -E '^U (exp|log|pow|sqrt)' undefined-sf_gfs.txt || true
    echo "# libm entry points module_gfs_funcphys.o calls:"
    grep -E '^U (exp|log|pow|sqrt)' undefined-funcphys.txt || true
} > libm-report.txt
cat libm-report.txt

# --- run ----------------------------------------------------------------------
python3 "${script_dir}/make_cases.py" cases.bin
mkdir -p out
./run_sf_gfs cases.bin out

if [[ -n "${wrf_build_root}" ]]; then
    # The same driver linked against the referee build's own objects.
    mkdir -p artefact
    # Object files only: the driver was compiled against this build's .mod
    # files, which describe the same interfaces, and linking resolves the
    # referee's symbols.
    for rel in "${units[@]}"; do
        cp "${wrf_build_root}/${rel%.F}.o" artefact/
    done
    gfortran -o run_sf_gfs_artefact run_sf_gfs.o gfs_seaice_wrapper.o \
        read_tbpvs.o "${objs[@]/#/artefact/}"
    mkdir -p out-artefact
    ./run_sf_gfs_artefact cases.bin out-artefact
    for f in out/*; do
        cmp "${f}" "out-artefact/$(basename "${f}")"
    done
    echo "artefact driver: every output byte-identical to the from-source driver"
fi

sha256sum cases.bin out/* > oracle-sha256sums.txt
for rel in "${units[@]}" phys/module_surface_driver.F; do
    sha256sum "${source_root}/${rel}" | sed "s#${source_root}#<wrf-4.6.1>#"
done >> oracle-sha256sums.txt
sha256sum "${script_dir}"/*.F90 "${script_dir}"/*.c "${script_dir}"/*.py \
    | sed "s#${script_dir}#tools/sfgfs_wrf461_oracle#" >> oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
