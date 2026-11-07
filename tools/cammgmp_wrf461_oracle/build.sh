#!/usr/bin/env bash
# Build and run the WRF v4.6.1 CAMMGMP (mp_physics = 11) column oracle.
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]
#
# The reference is the SCALAR WRF build's own libwrflib.a (the sprint's
# 0 ULP referee: -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize):
# the driver run_cammgmp.F90 calls CAM_INIT, CAMMGMP_INIT and CAMMGMP exactly
# as WRF does and links nothing of its own in their place.  Before linking,
# every CAM source the build compiled is compared byte for byte with the
# pinned pristine tree, so the archive is known to hold unmodified WRF.
#
# A second executable, run_cammgmp_fixed, links the same archive plus ONE
# recompiled module: phys/module_cam_mp_ndrop.F with the declared
# cloud-borne-number correction (fix_ndrop.py, same flags
# as the archive). This is an altered-source attribution instrument only.
# The stock kernel and pristine executable use unmodified WRF indexing.
# Generated soundings do not qualify as real forecast captures.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build=$(realpath "${3:-/work/wrf/build/scalar/WRF-4.6.1}")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
cam_sources=(phys/module_mp_cammgmp_driver.F phys/module_cam_mp_cldwat2m_micro.F
             phys/module_cam_mp_microp_aero.F phys/module_cam_mp_ndrop.F
             phys/module_cam_support.F phys/module_cam_physconst.F
             phys/module_cam_wv_saturation.F phys/module_cam_error_function.F
             phys/module_cam_constituents.F phys/module_cam_mp_qneg3.F
             phys/module_cam_mp_conv_water.F phys/module_cam_cldwat.F
             phys/module_cam_mp_modal_aero_initialize_data_phys.F
             phys/module_data_cam_mam_aero.F phys/module_cam_esinti.F
             phys/module_cam_gffgch.F phys/module_cam_shr_kind_mod.F
             phys/module_cam_shr_const_mod.F phys/module_physics_init.F)
if ! git -C "${source_root}" diff --quiet HEAD -- "${cam_sources[@]}"; then
    echo "a CAMMGMP dependency differs from pinned HEAD in ${source_root}" >&2
    exit 3
fi
for f in "${cam_sources[@]}"; do
    if ! cmp -s "${source_root}/${f}" "${wrf_build}/${f}"; then
        echo "${wrf_build}/${f} differs from the pinned source" >&2
        exit 3
    fi
done
test -f "${wrf_build}/main/libwrflib.a"

mkdir -p "${build_dir}"
cd "${build_dir}"

# The archive's own compile flags, read from the build's configure.wrf.
cat > print.mk <<EOF
include ${wrf_build}/configure.wrf
print:
	@echo "\$(LIB)"
optim:
	@echo "\$(FCOPTIM)"
cpp:
	@echo "\$(CPP) \$(CPPFLAGS)"
EOF
lib=$(make -s -f print.mk print WRF_SRC_ROOT_DIR="${wrf_build}")
optim=$(make -s -f print.mk optim WRF_SRC_ROOT_DIR="${wrf_build}")
if [[ "${optim}" != "-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize" ]]; then
    echo "unexpected FCOPTIM '${optim}' in ${wrf_build}" >&2
    exit 4
fi
inc=(-I"${wrf_build}/external/esmf_time_f90" -I"${wrf_build}/main"
     -I"${wrf_build}/external/io_netcdf" -I"${wrf_build}/external/io_int"
     -I"${wrf_build}/frame" -I"${wrf_build}/share" -I"${wrf_build}/phys"
     -I"${wrf_build}/inc")
fflags=(${optim} -w -ffree-form -ffree-line-length-none -fallow-argument-mismatch)

gfortran -c "${fflags[@]}" "${inc[@]}" -o run_cammgmp.o "${script_dir}/run_cammgmp.F90"
gcc -c -O2 -o dump_words.o "${script_dir}/dump_words.c"
# shellcheck disable=SC2086
gfortran -o run_cammgmp run_cammgmp.o dump_words.o "${wrf_build}/main/libwrflib.a" ${lib}

# Corrected-ndrop referee: the build's own preprocessed ndrop (the text the
# archive member was compiled from) with the declared fix, compiled with the
# archive's flags and linked ahead of it so its module replaces the member.
python3 "${script_dir}/fix_ndrop.py" "${wrf_build}/phys/module_cam_mp_ndrop.f90" \
    ndrop_fixed.f90
mkdir -p fixmod
gfortran -c "${fflags[@]}" -fconvert=big-endian -frecord-marker=4 -J fixmod \
    "${inc[@]}" -o ndrop_fixed.o ndrop_fixed.f90
gfortran -c "${fflags[@]}" -Ifixmod "${inc[@]}" -o run_cammgmp_fixed.o \
    "${script_dir}/run_cammgmp.F90"
# shellcheck disable=SC2086
gfortran -o run_cammgmp_fixed run_cammgmp_fixed.o ndrop_fixed.o dump_words.o \
    "${wrf_build}/main/libwrflib.a" ${lib}

# The -O2 archive must stay on scalar libm (no libmvec vector variants).
if nm -u run_cammgmp | grep -q '_ZGV'; then
    echo "the oracle links a libmvec vector symbol" >&2
    exit 5
fi

python3 "${script_dir}/make_cases.py" cases.bin
./run_cammgmp cases.bin pristine-out.bin helpers.bin words.txt
./run_cammgmp_fixed cases.bin fixed-out.bin helpers-fixed.bin words-fixed.txt
cmp helpers.bin helpers-fixed.bin
cmp words.txt words-fixed.txt

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# archive:  ${wrf_build}/main/libwrflib.a"
    echo "# FCOPTIM:  ${optim}"
} > toolchain.txt
sha256sum "${wrf_build}/main/libwrflib.a" "${script_dir}/run_cammgmp.F90" \
    "${script_dir}/make_cases.py" "${script_dir}/fix_ndrop.py" \
    "${script_dir}/dump_words.c" cases.bin pristine-out.bin fixed-out.bin \
    helpers.bin words.txt > oracle-sha256sums.txt
echo "${pinned_commit}" > wrf-commit.txt
echo "CAMMGMP oracle built in ${build_dir}"
