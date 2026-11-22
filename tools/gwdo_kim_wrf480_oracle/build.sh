#!/usr/bin/env bash
# The WRF v4.8.0 gwd_opt = 1 replay oracle, linked against the native strict
# referee's OWN object files.
#
#   build.sh REFEREE_480 REFEREE_471 INPUT_DIR OUT_DIR
#
# REFEREE_480 is the native build tree WRF-4.8.0-serial-pristine (commit
# 06d4240ae989cc3e50af412bb472df3d9048783c, the pinned strict recipe: GNU
# Fortran 13.3.0, glibc 2.39, FCOPTIM -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize; a development machine ~/wrf480-port/build).  REFEREE_471 is
# WRF-4.7.1-serial-pristine from the same recipe (commit f52c197ed39d...),
# for the two-sided replay of the frozen WRF v4.6.1/v4.7.1 kernel.  Run it
# inside the recipe's container (localhost/wrfref2404:1), whose gfortran and
# glibc the objects were built and are linked with:
#
#   podman run --rm -v $HOME/wrf480-port:/w -v $TREE:/t -v $DATA:/d \
#     localhost/wrfref2404:1 bash /t/tools/gwdo_kim_wrf480_oracle/build.sh \
#     /w/build/WRF-4.8.0-serial-pristine /w/build/WRF-4.7.1-serial-pristine \
#     /d/inputs /d/out
#
# The drivers (run_gwdo_kim480.F90 here, run_gwdo_kim.F90 of the v4.6.1
# oracle for 4.7.1) only read inputs and copy arrays; they are compiled -O0
# against the referee's .mod files and linked with its objects:
# phys/module_bl_gwdo.o, phys/physics_mmm/bl_gwdo.o, phys/ccpp_kind_types.o,
# share/module_model_constants.o.  OBJECTS.sha256 pins those objects' bytes
# (and the sources they were compiled from); any other bytes are refused.
# For every input set S in INPUT_DIR it writes OUT_DIR/gwdo-kim480-wrf480-S.bin
# (and gwdo-kim471-wrf471-S.bin), then oracle-sha256sums.txt and
# toolchain.txt.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: build.sh REFEREE_480 REFEREE_471 INPUT_DIR OUT_DIR" >&2
    exit 2
fi
r480=$(realpath "$1"); r471=$(realpath "$2"); inputs=$(realpath "$3")
out=$(realpath -m "$4")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gfv=$(gfortran --version | sed -n 1p)
case "$gfv" in *"13.3.0"*) ;; *)
    echo "REFUSED: the referee objects were built by GNU Fortran 13.3.0;" \
         "found '$gfv'" >&2; exit 3 ;; esac
check() {  # check TREE LABEL: the pinned objects and sources of that tree
    local tree=$1 label=$2
    ( cd "$tree" && grep "  ${label}:" "${here}/OBJECTS.sha256" \
        | sed "s/  ${label}:/  /" | sha256sum -c --quiet - ) || {
        echo "REFUSED: ${tree} differs from OBJECTS.sha256 (${label})" >&2
        exit 4; }
}
check "$r480" 480
check "$r471" 471
mkdir -p "$out/build480" "$out/build471"
link() {  # link TREE DRIVER EXE MODDIR
    local tree=$1 driver=$2 exe=$3 mods=$4
    gfortran -c -O0 -ffp-contract=off -cpp -ffree-form \
        -ffree-line-length-none -J "$mods" -I "$tree/phys" -I "$tree/frame" \
        -I "$tree/share" -o "$mods/driver.o" "$driver"
    gfortran -o "$exe" "$mods/driver.o" "$tree/phys/module_bl_gwdo.o" \
        "$tree/phys/physics_mmm/bl_gwdo.o" "$tree/phys/ccpp_kind_types.o" \
        "$tree/share/module_model_constants.o"
}
link "$r480" "$here/run_gwdo_kim480.F90" "$out/run_gwdo_kim480" "$out/build480"
link "$r471" "$here/../gwdo_kim_wrf461_oracle/run_gwdo_kim.F90" \
    "$out/run_gwdo_kim471" "$out/build471"
# The referee's own scheme objects call scalar libm only (no libmvec).
for obj in "$r480/phys/physics_mmm/bl_gwdo.o" "$r471/phys/physics_mmm/bl_gwdo.o"; do
    if nm -u "$obj" | grep -q '_ZGV'; then
        echo "REFUSED: $obj pulls a libmvec vector symbol" >&2; exit 5
    fi
done
shopt -s nullglob
n=0
for f in "$inputs"/gwdo-kim480-inputs-*.bin; do
    s=$(basename "$f" .bin); s=${s#gwdo-kim480-inputs-}
    "$out/run_gwdo_kim480" "$f" "$out/gwdo-kim480-wrf480-$s.bin"
    "$out/run_gwdo_kim471" "$inputs/gwdo-kim471-inputs-$s.bin" \
        "$out/gwdo-kim471-wrf471-$s.bin"
    n=$((n + 1))
done
[[ $n -gt 0 ]] || { echo "no input sets in $inputs" >&2; exit 6; }
{
    echo "# gfortran: $gfv"
    echo "# glibc:    $(ldd --version | sed -n 1p)"
    for obj in "$r480/phys/physics_mmm/bl_gwdo.o" "$r480/phys/module_bl_gwdo.o"; do
        echo "# $(basename "$obj") (4.8.0) libm imports: $(nm -u "$obj" \
            | awk '{print $2}' | grep -E 'f$' | paste -sd ' ')"
    done
} > "$out/toolchain.txt"
( cd "$here" && sha256sum build.sh run_gwdo_kim480.F90 make_inputs.py \
    compare_gpu.py OBJECTS.sha256 ) > "$out/oracle-sha256sums.txt"
( cd "$out" && sha256sum run_gwdo_kim480 run_gwdo_kim471 gwdo-kim4*-wrf4*-*.bin ) \
    >> "$out/oracle-sha256sums.txt"
( cd "$inputs" && sha256sum gwdo-kim4*-inputs-*.bin ) >> "$out/oracle-sha256sums.txt"
echo "gwdo-kim480 oracle: $n input set(s) replayed into $out"
