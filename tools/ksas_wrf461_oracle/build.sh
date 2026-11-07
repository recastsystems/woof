#!/usr/bin/env bash
# Build and run the KIAPS SAS (cu_physics = 14) column oracle against WRF
# v4.6.1.
#
#   build.sh WRF_SOURCE_ROOT WRF_SCALAR_BUILD_ROOT OUT_DIR [CASES_DIR]
#
# THE REFEREE IS THE SCALAR BUILD'S OWN OBJECT.  run_ksas links
# WRF_SCALAR_BUILD_ROOT/phys/module_cu_ksas.o, the object wrf.exe was linked
# from (-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize), so what is
# graded is the arithmetic the model runs, not a lookalike.  A second driver
# is linked against a copy compiled here from the PRISTINE source at the same
# flags; both must write identical bytes, or the build refuses (the breakage
# that prevents: an oracle silently graded against a build that is not the
# source it claims to be).
#
# The scheme module has no USE statements at all (no WRF framework, no
# constants module), so the file set is the one file.  `nm -u` on the object
# is the libm receipt: atanf, expf, powf -- tan(0.4*pi_) and log(100.) are
# folded at compile time and never reach libm.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: build.sh WRF_SOURCE_ROOT WRF_SCALAR_BUILD_ROOT OUT_DIR [CASES_DIR]" >&2
  exit 2
fi
src=$(realpath "$1"); bld=$(realpath "$2"); out=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cases=${4:-$out}
mkdir -p "$out/pristine" "$out/scalar"

# phys/module_cu_ksas.F at WRF v4.6.1 (tag d66e442); the same bytes sit in the
# scalar build tree.
pinned=84d88fe79c3aba7c299d0efaee216c596021c9659b445dd8d3365c5799a48854
for f in "$src/phys/module_cu_ksas.F" "$bld/phys/module_cu_ksas.F"; do
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [[ $got == "$pinned" ]] || { echo "refusing: $f sha256 $got != $pinned" >&2; exit 1; }
done

# configure.wrf (scalar): FCOPTIM + FCBASEOPTS_NO_G, verbatim.
FLAGS="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
# The WRF build preprocesses .F to .f90 with cpp -traditional before
# gfortran sees it (module_cu_ksas.F has no cpp directives, so the
# preprocessed text is the source).
/lib/cpp -P -nostdinc -traditional "$src/phys/module_cu_ksas.F" > "$out/pristine/module_cu_ksas.f90"
( cd "$out/pristine" && gfortran $FLAGS -c module_cu_ksas.f90 -o module_cu_ksas.o \
    && gfortran -O0 -ffree-line-length-none -c "$here/run_ksas.F90" -o run_ksas.o \
    && gfortran -o run_ksas run_ksas.o module_cu_ksas.o -lm )
cp "$bld/phys/module_cu_ksas.mod" "$out/scalar/"
( cd "$out/scalar" && gfortran -O0 -ffree-line-length-none -c "$here/run_ksas.F90" -o run_ksas.o \
    && gfortran -o run_ksas run_ksas.o "$bld/phys/module_cu_ksas.o" -lm )
nm -u "$bld/phys/module_cu_ksas.o" | awk '{print $2}' | sort > "$out/ksas-undefined-symbols.txt"
sha256sum "$bld/phys/module_cu_ksas.o" "$out/pristine/module_cu_ksas.o" > "$out/objects.sha256"

if [[ ! -f "$cases/cases.bin" ]]; then
  python3 "$here/gen_cases.py" --out "$cases"
fi
"$out/scalar/run_ksas" "$cases/cases.bin" "$out/ksas-oracle.bin"
"$out/pristine/run_ksas" "$cases/cases.bin" "$out/ksas-oracle-pristine.bin"
if ! cmp -s "$out/ksas-oracle.bin" "$out/ksas-oracle-pristine.bin"; then
  echo "refusing: the scalar build's object and the pristine compile disagree" >&2
  exit 1
fi
sha256sum "$cases/cases.bin" "$out/ksas-oracle.bin" | tee "$out/oracle.sha256"
echo "ksas oracle OK: $out/ksas-oracle.bin"
