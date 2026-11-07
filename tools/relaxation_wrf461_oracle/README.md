Selector 31 column oracle

Build from pristine WRF v4.6.1, commit d66e442fccc04111067e29274c9f9eaccc3cef28:

    bash tools/relaxation_wrf461_oracle/build.sh "$WRF_SOURCE_ROOT" "$L/oracle-build"

Through the GPU card queue, run both modes and the mutation control:

    GPUWM_WRF_EXACT=1 python tools/relaxation_wrf461_oracle/validate.py "$L/oracle-build/columns.hex" "$L/strict.json"
    python tools/relaxation_wrf461_oracle/validate.py "$L/oracle-build/columns.hex" "$L/default.json"
    GPUWM_WRF_EXACT=1 python tools/relaxation_wrf461_oracle/validate.py "$L/oracle-build/columns.hex" "$L/mutation.json" --mutate

The driver runs 64 columns with 8 levels, latitudes from -90 to 90 with an explicit zero-latitude column, sigma at 1, .95, .8, .70001, .7, .69999, .3 and .01. It includes nonzero initial rates, different adjacent pressure components, untouched halos and terminal faces. The complete WRF modules are compiled. Initialization is checked for both cold start and restart. The validator compares all output words, including untouched boundary faces. The one-bit mutation changes kkt inside the compiled kernel.

The runtime calls the generic radiation composition and shared no-FTZ loader. There is no scheme adapter file or coefficient table. The smoke runner integrates the real dycore and retains only full-field hashes. It is a dry local domain and does not substitute for the public global idealized initialization door or a matched WRF trajectory.
