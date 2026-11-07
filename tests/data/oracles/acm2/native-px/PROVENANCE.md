Real native ACM2/PX inputs and returned words
===========================================

These are sampled inputs and outputs from ACMPBL inside a strict native WRF
v4.7.1 forecast initialized from prepared real atmospheric data, not constructed
column inputs. The public-domain module is byte-identical to the pinned v4.6.1
module. WRF's public-domain licence and file pins are recorded in ../PROVENANCE.md
and licenses/LICENSE-WRF-public-domain.txt. No additional ACM2 tables are used.

The capture tool in tools/acm2_wrf461_oracle instruments the public native source.
Four MPI tiles supply 81 column instances at step 1, 49 levels, 12 seconds,
bl_pbl_physics=7, sf_sfclay_physics=7, sf_surface_physics=2, mp_physics=8,
ra_lw_physics=ra_sw_physics=4. The parent domain is 150x150 at 3 km, initialized
2024-05-25 18 UTC. Caller tendency inputs are zero in this capture. Stream words
are big-endian, as in WRF's own strict build. Gzip changes transport only.

manifest.json pins every raw and compressed file. Under GPUWM_NUMERICS=strict,
tests/test_acm2_native_px_replay_gpu.py unpacks the captured inputs and invokes
the shipped replay tool, comparing all 32,157 returned words with zero differences.
This module replay does not imply bit identity of the coupled forecast.
