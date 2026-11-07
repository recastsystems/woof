#!/usr/bin/env bash
# check_slab.sh: the mechanical pass/fail for the sf_surface_physics=1 port.
#
#   check_slab.sh TREE OUT_DIR
#
# Exit 0 only if all three hold:
#
#  (a) the oracle rebuilt from $WRF_SOURCE_ROOT reproduces the packaged
#      fixture byte for byte, and tests/test_slab_wrf461_oracle.py passes
#      under GPUWM_WRF_EXACT=1 (every output word of every admitted column,
#      every step, 0 ULP; a planted one-ULP change caught) and under the
#      default arithmetic (its per-field differences reported and pinned);
#  (b) `woof import-namelist` on WRF's own test/em_real/namelist.input.fire
#      no longer names sf_surface_physics (its mp_physics=2, fire and km_opt
#      refusals are other lanes' business);
#  (c) a 10-minute real 12 km forecast with sf_surface_physics=1 completes on
#      one card with finite TSK and TSLB in its history.
#
# Environment: WRF_SOURCE_ROOT (pristine WRF 4.6.1), SLAB_FIRE_WPS and
# SLAB_FIRE_INPUT (the namelist pair for (b)), SLAB_RUN_TOML and
# SLAB_RUN_DATA (the 12 km experiment and its staged forcing for (c)),
# SLAB_LANE (the card-queue lane name).  GPU steps go through the card
# queue, one card.
set -u
tree=$(realpath "${1:?usage: check_slab.sh TREE OUT_DIR}")
out=$(realpath -m "${2:?usage: check_slab.sh TREE OUT_DIR}")
mkdir -p "$out"
here="$tree/tools/slab_wrf461_oracle"
fixture="$tree/tests/data/oracles/slab"
lane=${SLAB_LANE:-pl-port-sf1-slab}
queue() {  # queue EST_SECONDS CMD...
    local est=$1; shift
    "${SLAB_GPU_QUEUE:?set SLAB_GPU_QUEUE to the shared GPU queue}" "$lane" --cards 1 --vram-mib "${SLAB_GPU_VRAM_MIB:-4000}" --est "$est" \
        --wait 3600 -- "$@"
}
status=0
verdict() { echo "$1: $2" | tee -a "$out/verdict.txt"; [[ $2 == PASS ]] || status=1; }
: > "$out/verdict.txt"

# ---- (a) oracle --------------------------------------------------------
if bash "$here/build.sh" "${WRF_SOURCE_ROOT:?}" "$out/oracle" > "$out/oracle-build.log" 2>&1 \
   && cmp -s "$out/oracle/slab-input.bin" "$fixture/slab-input.bin" \
   && cmp -s "$out/oracle/slab-output.bin" "$fixture/slab-output.bin"; then
    verdict "a.oracle-rebuild" PASS
else
    verdict "a.oracle-rebuild" FAIL
fi
env_py="T=$tree; source ${SLAB_ENV_SH:-/work/env.sh}; cd $tree"
if bash -c "$env_py; CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python -m pytest -q -m 'not gpu' tests/test_slab_wrf461_oracle.py tests/test_slab_integration.py" \
        > "$out/a-cpu.log" 2>&1; then
    verdict "a.cpu" PASS
else
    verdict "a.cpu" FAIL
fi
if queue 600 bash -c "$env_py; GPUWM_WRF_EXACT=1 python -m pytest -q -s -m gpu tests/test_slab_wrf461_oracle.py" \
        > "$out/a-gpu-strict.log" 2>&1 && grep -q "2 passed" "$out/a-gpu-strict.log"; then
    verdict "a.gpu-strict-0ulp" PASS
else
    verdict "a.gpu-strict-0ulp" FAIL
fi
if queue 600 bash -c "$env_py; python -m pytest -q -s -m gpu tests/test_slab_wrf461_oracle.py" \
        > "$out/a-gpu-default.log" 2>&1 && grep -q "2 passed" "$out/a-gpu-default.log"; then
    verdict "a.gpu-default-reported" PASS
else
    verdict "a.gpu-default-reported" FAIL
fi

# ---- (b) importer -------------------------------------------------------
bash -c "$env_py; python -m woof.cli import-namelist '${SLAB_FIRE_WPS:?}' '${SLAB_FIRE_INPUT:?}'" \
    > "$out/b-import.log" 2>&1
if ! grep -Eq "mp_physics|fire|Imported|imported" "$out/b-import.log" || grep -q "sf_surface_physics" "$out/b-import.log"; then
    verdict "b.import-no-sf_surface_physics" FAIL
else
    verdict "b.import-no-sf_surface_physics" PASS
fi

# ---- (c) real 12 km run -------------------------------------------------
# Reuse the one prepared bundle when it exists. The forecast still runs.
if [[ -n "${SLAB_PREPARED_ROOT:-}" ]]; then
    run_inputs="--prepared-root '$SLAB_PREPARED_ROOT' --wps-namelist '${SLAB_WPS_NAMELIST:?}'"
else
    run_inputs="--data-dir '${SLAB_RUN_DATA:?}'"
fi
if [[ -n "${SLAB_RUN_TOML:-}" ]] && queue 1800 bash -c "$env_py; python -m woof.cli go '$SLAB_RUN_TOML' $run_inputs --outdir '$out/run' --products none" \
        > "$out/c-run.log" 2>&1 \
   && bash -c "$env_py; python '$here/check_run_history.py' '$out/run'" \
        > "$out/c-history.log" 2>&1; then
    verdict "c.real-12km-10min" PASS
else
    verdict "c.real-12km-10min" FAIL
fi
exit $status
