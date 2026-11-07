# WDM7: native WRF conformance

The source is WRF v4.6.1 at `d66e442fccc04111067e29274c9f9eaccc3cef28`,
`phys/module_mp_wdm7.F`, distributed under WRF's public-domain statement.
`module_mp_radar.F`, `module_model_constants.F` and `libmassv.F` have the same provenance.
No external WDM tables are required.

The CUDA unit retains stock WRF rain transport, condensation limits, PLM index
updates, melting and number arithmetic. The prior modified reference arms are
retired. `GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1` selects replay numerics.
Default numerics permit FMA. Effective radii are stored in microns, matching
WRF radiation's FP32 multiplication by 1e6.

Two committed inputs were compared with the unmodified native module.
The synthetic diagnostic fixture has 91,080 output words and includes hail.
A capture from timestep two of a real native WRF forecast has 149 columns,
49 levels and a 15 second timestep, giving 110,856 output words. Both have
zero differences under strict numerics, including reflectivity and surface
precipitation. The real capture's incoming hail is zero; the synthetic hail
cases remain a separate diagnostic check.
`tests/data/oracles/wdm7/native_step2.provenance.json` records the input and
reference hashes. `tools/wdm_native_capture.py` reproduces the capture-only
instrumentation in a private pinned WRF build.

Column conformance does not establish forecast skill or whole-engine
performance. The lane evidence report records the real forecast, native and
observation maps, timings and bounded wiring checks. The registry keeps this
option at `wrf-matched-run-candidate`; it claims no independent scientific
validation.

Each CUDA thread owns one vertical column. Production launches reuse resident
scratch and CUDA preparation and finish kernels. The WDM6 translation units
are unchanged. State, IO and same-scheme nest paths carry CCN, cloud and rain
numbers, plus hail mass and hail surface accumulation.
Cross-scheme number and hail transitions remain refused pending qualification.
Native WRF reflectivity has no hail term; the port retains this behavior.
