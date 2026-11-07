"""Stock history bytes keep dry THM metadata; moist runs label the carrier."""
import netCDF4
import numpy as np
import pytest
from woof.io.wrfout import WrfoutWriter


@pytest.mark.parametrize("moist", [0, 1])
def test_history_thm_description_tracks_actual_carrier(tmp_path, moist):
    path = tmp_path / "wrfout_theta.nc"
    field = np.array([[[1.25, -0.5]]], dtype=np.float32)
    with WrfoutWriter(path, nx=2, ny=1, nz=1, dx=1000., dy=1000.,
                      global_attrs={"USE_THETA_M": moist}) as writer:
        writer.write_frame("2000-01-24_12:00:00", {"THM": field})
    with netCDF4.Dataset(path) as dataset:
        expected = ("perturbation moist potential temperature" if moist else
                    "perturbation dry potential temperature")
        assert dataset["THM"].description == expected
        assert dataset["THM"].units == "K"
        dataset["THM"].set_auto_maskandscale(False)
        np.testing.assert_array_equal(dataset["THM"][:].view(np.uint32), field[None].view(np.uint32))
