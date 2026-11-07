#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$source_root" ]]; then
    # Exported public sources carry no git metadata. The mandatory exact
    # module hash below binds this export to both audited upstream pins.
    module_source="$source_root"
    pin=f52c197ed39d12e087d02c50f412d90d418f6186
else
    pin=$(git -C "$source_root" rev-parse HEAD)
    case "$pin" in
    d66e442fccc04111067e29274c9f9eaccc3cef28|f52c197ed39d12e087d02c50f412d90d418f6186) ;;
    *) echo 'WRF source must be the pinned v4.6.1 or v4.7.1 commit' >&2; exit 2 ;;
    esac
    git -C "$source_root" diff --exit-code HEAD -- phys/module_ra_gsfcsw.F
    module_source="$source_root/phys/module_ra_gsfcsw.F"
fi
echo 'da257cf060c385c3aa4d6c3a737ec8c10773bd52034786d954bab0a0dc290766  '"$module_source" | sha256sum --check
mkdir -p "$build_dir"
cd "$build_dir"
flags=(-O0 -fPIC -ffp-contract=off -fno-tree-vectorize -cpp -ffree-form -ffree-line-length-none -DWRF_CHEM=0)
gfortran "${flags[@]}" -c "$module_source" -o wrf.o
gfortran "${flags[@]}" -c "$script_dir/column.F90" -o column.o
gfortran -shared wrf.o column.o -o oracle.so
nm -u wrf.o > undefined.txt
if grep -q _ZGV undefined.txt; then echo 'scalar oracle called vector math'; exit 3; fi
sha256sum "$module_source" "$script_dir/column.F90" oracle.so > sha256sums.txt
gfortran --version | head -1 > compiler.txt
getconf GNU_LIBC_VERSION > libc.txt
printf '%s\n' "$pin" > wrf-commit.txt
"${PYTHON:-python3}" "$script_dir/record.py" "$build_dir"
