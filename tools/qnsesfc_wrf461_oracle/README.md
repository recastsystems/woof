# QNSE surface layer

`phys/module_sf_qnsesfc.F` is from WRF's official public source. Its
v4.7.1 and v4.6.1 bytes are identical. `source-pins.json` lists official
URLs, hashes and sizes; `LICENSE-WRF.txt` records the public-domain
licence. No separately restricted table is required.

QNSE is selected by `sf_sfclay_physics=4` paired with
`bl_pbl_physics=4`. The shared Eta/QNSE launcher dispatches the QNSE CUDA
kernel and its own QNSESFCINIT tables. The two frozen table generations
are read through Rust NetCDF once and uploaded to the GPU once.
`make_tables.py` is an offline regeneration tool, outside the step loop.

Production uses fast CUDA arithmetic. Strict word comparison requires
`GPUWM_WRF_EXACT=1`. The synthetic fixture covers replay and chained
calls; real native-run capture replay is implemented in
`../qnse_wrf461_oracle/replay_native.py`. Driver frame measurements are
recorded under `receipts/`; compile-only measurements are historical.
