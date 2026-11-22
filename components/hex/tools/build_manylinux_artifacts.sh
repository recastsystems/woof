#!/usr/bin/env bash
# Build the release wheel and sdist exactly as .github/workflows/publish.yml's
# build job does, on a host with podman or docker: same pinned manylinux_2_28
# image, same toolchain, sdist first and the wheel from the sdist, auditwheel
# repair, then tools/check_wheel_platform.py.  This is the "publish.yml-
# equivalent container" a cut builds in before anything is pushed.
#
#   tools/build_manylinux_artifacts.sh ENGINE_SOURCE OUT_DIR
#
# ENGINE_SOURCE is a woof source tree at the declared floor's tag (tools/rustwx,
# tools/grib1_bridge/vendor and its pyproject.toml), which the observation
# scorer links.  OUT_DIR receives dist/, SHA256SUMS, dist-check.json and build.log.
# WOOF_HEX_ENGINE_BRIDGES, when set, names the directory holding the pinned
# rw_mpas_geometry, rw_mpas_stream and rw_mpas_static (tools/build_engine_bridges.sh; the release
# downloads them from the assets src/hexcore/data/engine-bridges.json names).
set -euo pipefail
engine=$(cd -- "${1:?ENGINE_SOURCE}" && pwd)
out=$(mkdir -p -- "${2:?OUT_DIR}" && cd -- "$2" && pwd)
tree=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workflow="$tree/../.github/workflows/publish.yml"
image=$(sed -n 's/^ *image: *\(quay.io\/pypa\/manylinux_2_28_x86_64@sha256:[0-9a-f]*\).*/\1/p' "$workflow")
toolchain=$(sed -n 's/^ *RUST_TOOLCHAIN: *"\([^"]*\)".*/\1/p' "$workflow")
policy=$(sed -n 's/^ *MANYLINUX_POLICY: *\([a-z0-9_]*\).*/\1/p' "$workflow")
requires=$(grep -o -- '--require-libexec [A-Za-z0-9_-]*' "$workflow" | sort -u | tr '\n' ' ')
[ -n "$image" ] && [ -n "$toolchain" ] && [ -n "$policy" ] || { echo "cannot read image/toolchain/policy from $workflow" >&2; exit 2; }
engine_cli=$(command -v podman || command -v docker) || { echo "needs podman or docker" >&2; exit 2; }
netcdf_line=""
bridges_mount=() bridges_line=""
if [ -n "${WOOF_HEX_ENGINE_BRIDGES:-}" ]; then
  bridges_mount=(-v "$(cd -- "$WOOF_HEX_ENGINE_BRIDGES" && pwd):/engine-bridges:ro,Z")
  bridges_line="export WOOF_HEX_ENGINE_BRIDGES=/engine-bridges"
fi
grep -q "build_netcdf_classic_static.sh" "$workflow" && netcdf_line="bash /src/tree/tools/build_netcdf_classic_static.sh /opt/netcdf-classic; export LIBRARY_PATH=/opt/netcdf-classic/lib"
cat >"$out/job.sh" <<EOF
#!/bin/bash
set -euo pipefail
PY=/opt/python/cp311-cp311/bin/python
${netcdf_line}
${bridges_line}
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${toolchain}" >/dev/null
export PATH="\$HOME/.cargo/bin:\$PATH"
cp -a /src/tree /build-tree
cd /build-tree
"\$PY" -m pip install -q --upgrade pip build
"\$PY" -m pip install -q --target /opt/hexad-nvcc nvidia-cuda-nvcc==13.5.37 nvidia-cuda-crt==13.5.37 nvidia-cuda-runtime==13.5.37 nvidia-nvvm==13.5.37
export NVCC=/opt/hexad-nvcc/nvidia/cu13/bin/nvcc GPUWM_HEX_REQUIRE_AD_FATBINS=1
GPUWM_HEX_ENGINE_SOURCE=/engine "\$PY" -m build --outdir /out/raw
cd /out
rm -rf dist && mkdir dist
auditwheel repair --plat "${policy}" --wheel-dir dist raw/*.whl
cp raw/*.tar.gz dist/
"\$PY" /src/tree/tools/check_wheel_platform.py dist/*.whl --policy "${policy}" ${requires} --json dist-check.json
(cd dist && sha256sum *) > SHA256SUMS
rm -rf raw
EOF
chmod +x "$out/job.sh"
"$engine_cli" run --rm -v "$tree/..:/src:ro,Z" -v "$engine:/engine:ro,Z" "${bridges_mount[@]}" -v "$out:/out:Z" "$image" /out/job.sh 2>&1 | tee "$out/build.log"
cat "$out/SHA256SUMS"
