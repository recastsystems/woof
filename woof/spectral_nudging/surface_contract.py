"""Surface FDDA registry rows owned beside their runtime consumer."""
def registry_rows():
    from .surface_forecast import HELD, FASDAS_HELD
    common = {
        "implemented": True, "maturity": "experimental-runtime",
        "namelist": "fdda.grid_sfdda", "requires": {"grid_fdda": 1},
        "consumer": "woof.spectral_nudging.surface_forecast.SurfaceForecast",
        "inputs": "native auxinput9 OLD/NEW endpoints; Rust bounded parallel stored-word reads",
        "restart": "serialized tendencies and fluxes; rebuilt per-domain/member device clock; input SHA256 bound by native input authority",
        "execution": "resident regional domains; CUDA columns, cadence and C-grid RK coupling; both nest feedback modes",
        "upstream": "WRF v4.6.1 phys/module_fdda_psufddagd.F SFDDAGD",
        "proof": "woof/spectral_nudging/qualified-proof.json",
    }
    return {
        "grid_sfdda": {
            "0": {"implemented": True, "behavior": "allocation-free off path"},
            "1": {**common, "state": [*HELD, "hfx_fdda"]},
            "2": {**common, "state": [*HELD, "hfx_fdda", *FASDAS_HELD],
                  "land_coupling": "Noah native evaporation partition, QNORM and first soil-layer heat correction",
                  "native_undefined": "Noah glacier HCPCT_FASDAS is never assigned by SFLX_GLACIAL; those columns are refused with the named native-undefined reason"}},
        "px_auxinput9": {
            "implemented": True, "forecast_qualified": True,
            "consumer": "woof.spectral_nudging.px_inputs.PxInputs",
            "inputs": "native T2/Q2/SN OLD/NEW endpoints; bounded Rust reads and resident CUDA record selection",
            "clock": "per-domain/member CUDA clock reconstructed from checkpoint elapsed time",
            "proof": "woof/data/pxlsm/soil-nudging-proof.json"}}
