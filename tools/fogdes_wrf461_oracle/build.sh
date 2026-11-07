#!/usr/bin/env bash
# Build and run the FogDES column oracle (grav_settling = 1, 2) against
# WRF v4.6.1.
#
#   bash tools/fogdes_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]
#
# WRF_SOURCE_ROOT holds the pristine v4.6.1 tree; the three files
# SOURCES.sha256 pins are checked before anything compiles, so a fixture can
# never come from an edited file.  Compiled byte-unmodified:
#
#   share/module_model_constants.F   g = 9.81, rhowater = 1000.
#   phys/module_bl_fogdes.F          bl_fogdes (settling flux into RQCBLTEN)
#   phys/module_sf_fogdes.F          sf_fogdes (VDFG, DFGDP, FGDP)
#
# plus stub_wrf.F90 (the error entry points), ../urban_wrf471_oracle/
# oracle_io.F90 (the fixture writer) and run_fogdes.F90 (the cases).
#
# Three arms, which must write byte-identical fixtures or the build fails:
#
#   O0      the WRF sources at -O0;
#   wrfopt  the WRF sources at the flags the scalar referee wrf.exe was built
#           with (configure.wrf FCOPTIM: -O2 -fno-fast-math -ffp-contract=off
#           -fno-tree-vectorize), where gfortran lowers X**2.0 to X*X and
#           keeps every other REAL power a powf call;
#   wrfexe  (only with WRF_BUILD_ROOT) the driver linked against that build's
#           OWN objects -- the module_bl_fogdes.o, module_sf_fogdes.o and
#           module_model_constants.o inside its main/wrf.exe.
#
# The fixture published is O0's; BUILD_DIR/fixtures, copied to
# woof/data/fogdes/oracle/.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build=${3:+$(realpath "$3")}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf O0 wrfopt wrfexe fixtures

ff=(-ffree-form -ffree-line-length-none)
wrfopt=(-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize)

build_arm() {
    local arm=$1; shift
    local flags=("$@")
    mkdir -p "${arm}"
    (
        cd "${arm}"
        for src in share/module_model_constants.F phys/module_bl_fogdes.F \
                   phys/module_sf_fogdes.F; do
            gfortran -c "${flags[@]}" -cpp -DEM_CORE=1 "${ff[@]}" \
                "${source_root}/${src}"
        done
        for obj in module_bl_fogdes.o module_sf_fogdes.o; do
            if nm -u "${obj}" | grep -q '_ZGV'; then
                echo "${arm}: ${obj} pulled in a libmvec vector symbol" >&2
                exit 4
            fi
            echo "# ${arm} ${obj}: $(nm -u "${obj}" | grep -cw powf) powf" \
                 "import(s); calls: $(objdump -dr "${obj}" | grep -c 'R_X86_64_PLT32.*powf')" \
                 >> ../libm-report.txt
        done
        gfortran -c -O0 "${ff[@]}" "${script_dir}/stub_wrf.F90"
        gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
        gfortran -c -O0 "${ff[@]}" "${script_dir}/run_fogdes.F90"
        gfortran -o run_fogdes run_fogdes.o module_bl_fogdes.o \
            module_sf_fogdes.o module_model_constants.o stub_wrf.o oracle_io.o
        mkdir -p fixtures
        ./run_fogdes fixtures
    )
}

: > libm-report.txt
build_arm O0 -O0
build_arm wrfopt "${wrfopt[@]}"
diff -r O0/fixtures wrfopt/fixtures > /dev/null || {
    echo "the -O0 and WRF-flag arms disagree" >&2
    exit 6
}

if [[ -n "${wrf_build}" ]]; then
    mkdir -p wrfexe
    (
        cd wrfexe
        for obj in phys/module_bl_fogdes.o phys/module_sf_fogdes.o \
                   share/module_model_constants.o; do
            cp "${wrf_build}/${obj}" .
        done
        # The driver compiles against O0's module interfaces (the same
        # source) and links the referee build's own objects.
        cp ../O0/*.mod .
        gfortran -c -O0 "${ff[@]}" "${script_dir}/stub_wrf.F90"
        gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
        gfortran -c -O0 "${ff[@]}" "${script_dir}/run_fogdes.F90"
        gfortran -o run_fogdes run_fogdes.o module_bl_fogdes.o \
            module_sf_fogdes.o module_model_constants.o stub_wrf.o oracle_io.o
        mkdir -p fixtures
        ./run_fogdes fixtures
        sha256sum module_bl_fogdes.o module_sf_fogdes.o module_model_constants.o \
            > OBJECTS.sha256
    )
    diff -r O0/fixtures wrfexe/fixtures > /dev/null || {
        echo "the referee wrf.exe objects disagree with the source build" >&2
        exit 7
    }
    echo "# wrfexe: ${wrf_build} objects agree byte for byte" >> libm-report.txt
    sed 's/^/# /' wrfexe/OBJECTS.sha256 >> libm-report.txt
fi

cp -r O0/fixtures fixtures
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
    echo "arms: O0 wrfopt${wrf_build:+ wrfexe} (byte-identical)"
} > fixtures/TOOLCHAIN.txt
cp libm-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh run_fogdes.F90 stub_wrf.F90 \
    ../urban_wrf471_oracle/oracle_io.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
