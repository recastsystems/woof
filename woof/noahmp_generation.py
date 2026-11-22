"""Numerical generations for the Noah-MP snow-aging call placement.

This selector changes only call placement. It does not select the complete
Noah-MP 5.1.2 model shipped with WRF v4.8.0.
"""

NOAHMP_SNOW_AGE_GENERATIONS = ("wrf_470", "wrf_461")

# The NCAR/noahmp v4.x commit each generation reproduces (the permissive WRF-lineage line; v5.x is UCAR-licensed and
# never read).  The default, wrf_470, is e5c0859 (the submodule of WRF v4.7.1, 2025-04-11), the newest v4.x commit in a
# WRF release.  Of its changes since 848f54ad only 20ba768's night-time snow aging (lsm and glacier) touches ArWen's
# supported options (TAUSS, and BATS albedo at opt_alb=1); the others are the opt_sfc=2 stability functions, the
# irrigation-table initialisation, the OpenMP soiltstep flag and the glacier HCPCT the port already defines, none of
# which a supported option reaches.  (release-v4.8-WRF 1d4d953 adds only urban NBS plumbing; no WRF release uses it.)
# wrf_461 is 848f54ad, the WRF v4.6.1 submodule the bitwise referee pins (core/noahmp_mynn_contract.py).
NOAHMP_V4_COMMITS = {
    "wrf_470": "e5c0859874407859936739e8be8741f9aed369ee",
    "wrf_461": "848f54ad3d28c4303151fe5ad83724e232694422",
}


def snow_age_generation(value):
    if value not in NOAHMP_SNOW_AGE_GENERATIONS:
        raise ValueError(
            f"noahmp_snow_age={value!r} must be 'wrf_470' (night aging) or "
            "'wrf_461' (daylight-only reproduction); substituting a generation "
            "would change TAUSS and the following dawn's BATS albedo")
    return value


def radiation_kernel(value):
    return ("noahmp_radiation_batch_night_age" if
            snow_age_generation(value) == "wrf_470" else "noahmp_radiation_batch")
