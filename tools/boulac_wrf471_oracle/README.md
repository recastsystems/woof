# BouLac native WRF 4.7.1 capture

The source and licence URLs and SHA-256 values are in PIN.json. The WRF
public-domain statement covers module_bl_boulac.F; no external tables or
restricted source are shipped by this port. WRF is supplied without warranty.
Credit the WRF model when publishing results, as its licence requests.

Copy a strict native build into lane scratch before running instrument_native.py
on its phys/module_bl_boulac.F. Rebuild with at most eight jobs. The tap writes
the first real module call on each MPI tile to boulac-I-J.in.bin and .rec.bin.
The tap changes no arguments or physics expressions. Preserve the original
and instrumented executable and module hashes in the run receipt.

Replay with GPUWM_WRF_EXACT=1 through woof.core.boulac.launch_boulac.
The binary layout is the layout read by woof.verify.boulac_oracle. Compare
all tendency, TKE, length, diffusivity, flux and PBLH words. Rural DLG_BEP
is not read or written by the scheme and is excluded from real-input grading.
The calm, zero-stress native U/V NaN case must be counted explicitly.

The older tools/boulac_wrf461_oracle produces generated-column fixtures.
Its module source hash equals the v4.7.1 hash; these fixtures remain a
separate regression check and do not establish real-forecast validation.
