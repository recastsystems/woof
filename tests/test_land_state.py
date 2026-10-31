"""woof/ingest/land_state.py: a foreign soil model's moisture is index-remapped, a donor is land-only.

Named breakage (found 2026-10-26): an ERA5 start copied
HTESSEL soil moisture into RUC as is (0.36 against HRRR's 0.21) and T2 ran
1-2 C cold from the first hour.
"""
import numpy as np

from woof.ingest import land_state

_ERA5 = ("SM000007", "SM007028", "SM028100", "SM100289")


def _era5(value):
    return {name: np.full((2, 3), value) for name in _ERA5}


def test_era5_moisture_reaching_ruc_is_index_remapped_not_copied():
    soil_type = np.array([[4, 6, 8], [3, 12, 9]])  # STAS loams, clay, sandy loam
    fields = _era5(0.36)
    out, receipt = land_state.convert_source_soil_moisture(
        fields, sf_surface_physics=3, soil_type=soil_type)
    assert receipt["source_soil_model"] == "htessel" and receipt["target_soil_model"] == "ruc"
    assert out is not fields and receipt["land_cells_converted"] == 6
    # at HTESSEL's medium field capacity (0.347) a loam sits at RUC's own
    # field capacity, not at the HTESSEL number
    loam = out["SM000007"][0, 0]
    from woof.core.noah import load_tables
    ruc = load_tables(mminsl="STAS-RUC")
    wp, fc = ruc.wltsmc[3], ruc.refsmc[3]
    expected = wp + (0.36 - 0.151) / (0.347 - 0.151) * (fc - wp)
    assert np.isclose(loam, expected)


def test_same_soil_model_and_opt_out_pass_the_source_through(monkeypatch):
    soil_type = np.full((2, 3), 6)
    gfs = {f"GFS_SM{a}": np.full((2, 3), 0.3) for a in ("000010", "010040", "040100", "100200")}
    out, receipt = land_state.convert_source_soil_moisture(gfs, sf_surface_physics=2, soil_type=soil_type)
    assert out is gfs and receipt["moisture"].startswith("copied")
    monkeypatch.setenv("WOOF_SOIL_LAND_STATE", "off")
    fields = _era5(0.36)
    out, receipt = land_state.convert_source_soil_moisture(fields, sf_surface_physics=3, soil_type=soil_type)
    assert out is fields and "off" in receipt["moisture"]


def test_hrrr_ruc_moisture_reaching_noah_is_copied_not_remapped():
    """Named breakage (WOOF NYC 2026-10-28 12Z, Noah city templates): the index
    remap moved HRRR's RUC moisture onto Noah's lower wilting points (natural
    top layer 0.150 -> 0.106), latent heat fell to 40 W/m2 against HRRR's 67
    and the 2 m dew point ran dry against ASOS.  RUC into Noah copies, as
    stock WRF does; HTESSEL into Noah or RUC still remaps."""
    soil_type = np.array([[4, 6, 8], [3, 12, 9]])
    ruc = {"SOILW": np.full((9, 2, 3), 0.17, dtype=np.float32),
           "SOILT": np.full((9, 2, 3), 285.0, dtype=np.float32)}
    for scheme in (2, 4):  # Noah, Noah-MP
        out, receipt = land_state.convert_source_soil_moisture(
            ruc, sf_surface_physics=scheme, soil_type=soil_type)
        assert receipt["source_soil_model"] == "ruc" and receipt["target_soil_model"] == "noah"
        assert out is ruc and receipt["moisture"].startswith("copied: the table's copy pair")
    out, receipt = land_state.convert_source_soil_moisture(
        _era5(0.36), sf_surface_physics=2, soil_type=soil_type)
    assert receipt["moisture"] == "soil-moisture index remap"


def test_donor_sampling_takes_land_only_and_searches_the_nearest_land():
    values = np.full((1, 4, 4), 1.0, dtype=np.float32)   # water = 1.0
    values[0, :, :2] = 0.2                                 # land on the west half
    land = values[0] < 1.0
    x = np.array([1.5, 3.0, 9.0]); y = np.array([1.0, 1.0, 1.0])
    out, found = land_state._sample_land(values, land, x, y, radius=3)
    assert found.tolist() == [True, True, False]
    assert np.allclose(out[0, :2], 0.2)                    # never mixed with the water 1.0
