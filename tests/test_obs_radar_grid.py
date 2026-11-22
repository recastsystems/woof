"""``gpuwm-obs.radar-grid.v1``: round-trip, identity binding, and refusals.

The schema is a cross-lane contract, so the tests are about what a DA lane
can rely on: that every named variable survives a write/read cycle
bit-for-bit at float32, that the file is welded to a grid identity, and
that a mismatch is an exception rather than a silent regrid.

The synthetic volume here is built in memory rather than decoded from a
Level-II file, so the superob arithmetic is checked against numbers that
can be worked out by hand: one radar, a handful of gates, known dBZ and
known velocities.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.obs.radar_grid import (RADAR_GRID_SCHEMA, RADAR_GRID_STATUS,
                                  RadarGridSchemaError, read_radar_grid,
                                  write_radar_grid)
from woof.obs.sweeps import Moment, RadarSite, RadarVolume, Sweep
from woof.obs.superob import (SuperobParams, merge_contributions,
                               superob_volume)
from woof.obs.target_grid import GridMismatchError, TargetGrid
from woof.static.lambert import LambertGrid


def _grid(nx: int = 41, ny: int = 41, dx: float = 2000.0, nz: int = 10,
          top_m: float = 10000.0) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, top_m, nz + 1), name="analytic")


def _volume(grid: TargetGrid, *, reflectivity, velocity, nyquist=32.0,
            azimuths=(90.0,), elevation=0.5, gate_size=250.0,
            first_gate=2125.0, site_id="KTLX") -> RadarVolume:
    """A synthetic volume centred on the grid, with the values given.

    ``reflectivity`` and ``velocity`` are per-gate arrays broadcast over
    the azimuths, so a caller controls exactly which numbers land where.
    """

    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    azimuth = np.asarray(azimuths, dtype=np.float32)
    def plane(values):
        values = np.asarray(values, dtype=np.float32)
        return np.tile(values[None, :], (azimuth.size, 1))

    moments = {}
    if reflectivity is not None:
        data = plane(reflectivity)
        moments["REF"] = Moment("REF", "dBZ", data.shape[1], first_gate,
                                gate_size, data)
    if velocity is not None:
        data = plane(velocity)
        moments["VEL"] = Moment("VEL", "m/s", data.shape[1], first_gate,
                                gate_size, data)
    sweep = Sweep(
        sweep_index=0, elevation_number=1, elevation_angle_deg=elevation,
        nyquist_velocity_ms=nyquist, start_status=3, end_status=2,
        cut_sector=0, complete=True,
        azimuth_deg=azimuth,
        elevation_deg=np.full(azimuth.size, elevation, dtype=np.float32),
        moments=moments)
    return RadarVolume(
        site=RadarSite(id=site_id, name="synthetic",
                       lat_deg=float(grid.lat[centre_j, centre_i]),
                       lon_deg=float(grid.lon[centre_j, centre_i]),
                       alt_m=0.0, source="test"),
        valid_time="2026-07-28T20:03:16Z", station_id=site_id,
        volume_file=f"{site_id}20260728_200316_V06",
        volume_sha256="0" * 64, volume_bytes=8102058,
        pack_path=__import__("pathlib").Path("synthetic.rdrpack"),
        pack_sha256="1" * 64,
        params={"moments": ["REF", "VEL"], "max_range_km": 250.0},
        framing={"magic": "AR2V0006", "block_count": 1},
        sweeps=(sweep,))


def _gridded(grid, volume, params=None, z_reduce="max"):
    # ``dealias=None`` is STATED here, not inherited.  Dealiasing is what a
    # bare SuperobParams does now, and every assertion below this line is
    # about the masking-only path: what the four risk masks drop, what they
    # count, and the statement the file makes when nothing was unfolded.
    # Leaving it to the default would silently measure a different pass.
    params = params or SuperobParams(dealias=None)
    contribution = superob_volume(volume, grid, params=params)
    return merge_contributions([contribution], grid, params=params,
                               z_reduce=z_reduce), params


def test_radar_grid_round_trips_every_named_variable(tmp_path):
    grid = _grid()
    gates = 40
    volume = _volume(grid, reflectivity=np.linspace(5.0, 55.0, gates),
                     velocity=np.linspace(-20.0, 20.0, gates))
    observations, params = _gridded(grid, volume)
    path = tmp_path / "radar-grid.nc"

    receipt = write_radar_grid(path, observations, grid,
                               valid_time="2026-07-28T20:03:16Z",
                               params=params)
    assert receipt["schema"] == RADAR_GRID_SCHEMA
    assert receipt["status"] == RADAR_GRID_STATUS
    assert receipt["grid_identity_sha256"] == grid.identity_sha256()
    assert receipt["dims"] == {"level": grid.nz, "south_north": grid.ny,
                               "west_east": grid.nx, "radar": 1}
    assert receipt["bytes"] == path.stat().st_size

    read = read_radar_grid(path, expected_grid_identity=grid.identity_sha256())
    assert read["schema"] == RADAR_GRID_SCHEMA
    assert read["valid_time"] == "2026-07-28T20:03:16Z"
    assert read["dims"] == receipt["dims"]
    assert read["radars"][0]["id"] == "KTLX"
    assert read["radars"][0]["valid_time"] == "2026-07-28T20:03:16Z"
    assert read["radars"][0]["lat_deg"] == pytest.approx(volume.site.lat_deg)
    assert read["superob_params"]["nyquist_reject_fraction"] == pytest.approx(
        params.nyquist_reject_fraction)

    for name, expected in (
            ("z_obs", observations.z_obs), ("z_err", observations.z_err),
            ("z_max", observations.z_max), ("z_mean", observations.z_mean),
            ("vr_obs", observations.vr_obs), ("vr_err", observations.vr_err),
            ("vr_beam_east", observations.vr_beam_east),
            ("vr_beam_north", observations.vr_beam_north),
            ("vr_beam_up", observations.vr_beam_up)):
        assert np.array_equal(read["variables"][name],
                              np.asarray(expected, dtype=np.float32)), name
    for name, expected in (("z_mask", observations.z_mask),
                           ("vr_mask", observations.vr_mask),
                           ("z_count", observations.z_count),
                           ("vr_count", observations.vr_count),
                           ("vr_rejected", observations.vr_rejected)):
        assert np.array_equal(read["variables"][name], expected), name

    # The provenance carries the volume that produced it.
    volumes = read["provenance"]["volumes"]
    assert volumes[0]["volume_file"] == "KTLX20260728_200316_V06"
    assert volumes[0]["pack_schema"] == "gpuwm-obs.radar-sweeps.v1"
    assert read["provenance"]["counts"][0]["gates_considered"] > 0


def test_reader_fails_closed_on_a_grid_it_was_not_written_for(tmp_path):
    grid = _grid()
    volume = _volume(grid, reflectivity=np.full(20, 30.0), velocity=None)
    observations, params = _gridded(grid, volume)
    path = tmp_path / "radar-grid.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params)

    other = _grid(nz=12, top_m=12000.0)
    assert other.identity_sha256() != grid.identity_sha256()
    with pytest.raises(GridMismatchError, match="is bound to grid"):
        read_radar_grid(path, expected_grid_identity=other.identity_sha256())
    # Without a demand it still reads: fail-closed is the caller's opt-in.
    assert read_radar_grid(path)["grid_identity_sha256"] == \
        grid.identity_sha256()


def test_writer_refuses_a_shape_that_is_not_the_target_grid(tmp_path):
    grid = _grid()
    volume = _volume(grid, reflectivity=np.full(20, 30.0), velocity=None)
    observations, params = _gridded(grid, volume)
    observations.z_obs = observations.z_obs[:-1]
    with pytest.raises(GridMismatchError, match="z_obs has shape"):
        write_radar_grid(tmp_path / "bad.nc", observations, grid,
                         valid_time="2026-07-28T20:03:16Z", params=params)
    assert not (tmp_path / "bad.nc").exists()


def test_writer_refuses_to_overwrite_and_leaves_no_temp_behind(tmp_path):
    grid = _grid()
    volume = _volume(grid, reflectivity=np.full(20, 30.0), velocity=None)
    observations, params = _gridded(grid, volume)
    path = tmp_path / "radar-grid.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_radar_grid(path, observations, grid,
                         valid_time="2026-07-28T20:03:16Z", params=params)
    assert not list(tmp_path.glob(".*tmp*"))


def test_reader_refuses_a_file_that_is_not_this_schema(tmp_path):
    import netCDF4

    path = tmp_path / "impostor.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.schema = "gpuwm-obs.radar-grid.v0"
    with pytest.raises(RadarGridSchemaError, match="schema"):
        read_radar_grid(path)

    path = tmp_path / "no-identity.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.schema = RADAR_GRID_SCHEMA
    with pytest.raises(RadarGridSchemaError, match="grid_identity_sha256"):
        read_radar_grid(path)


def _rewrite(source, destination, *, mutate):
    """Copy a radar-grid file, letting ``mutate`` change it on the way.

    ``mutate(dataset_in, dataset_out)`` owns creating dimensions and
    variables in the destination; everything it does not create is copied
    verbatim, attributes included.  This is how a file that keeps a
    *legitimate* identity attribute while changing what sits under it gets
    built -- which is exactly the attack RAD-H2 described.
    """

    import netCDF4

    with netCDF4.Dataset(source, "r") as src, \
            netCDF4.Dataset(destination, "w",
                            format="NETCDF4_CLASSIC") as dst:
        dst.setncatts({key: src.getncattr(key) for key in src.ncattrs()})
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, len(dimension))
        made = mutate(src, dst)
        for name, variable in src.variables.items():
            if name in made:
                continue
            out = dst.createVariable(name, variable.dtype,
                                     variable.dimensions)
            out.setncatts({key: variable.getncattr(key)
                           for key in variable.ncattrs()
                           if key != "_FillValue"})
            out[:] = variable[:]
    return destination


@pytest.fixture(scope="module")
def _square_grid_file(tmp_path_factory):
    """A 2x3x3 file: square, so a transpose keeps every array shape."""

    tmp_path = tmp_path_factory.mktemp("square-radar-grid")
    grid = _grid(nx=3, ny=3, dx=100000.0, nz=2, top_m=8000.0)
    volume = _volume(grid, reflectivity=[30.0, 35.0], velocity=[5.0, 6.0],
                     nyquist=32.0)
    observations, params = _gridded(grid, volume)
    path = tmp_path / "square.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params)
    return path, grid


def test_a_transposed_or_staggered_variable_is_refused_not_regridded(tmp_path, _square_grid_file):
    """RAD-H2, rebuilt from the audit's own adversarial construction.

    A square 2x3x3 file whose ``z_obs`` is declared over
    ``('level', 'west_east', 'south_north_stag')`` -- a transpose and an
    equal-length stagger substitution at once -- while the file keeps its
    real ``grid_identity_sha256``.  Array shape, rank and dtype are all
    correct; the values are on the wrong axes and half a cell out.  The
    reader used to accept this and the DA adapter returned a (2, 3, 3)
    reflectivity batch from it.
    """

    import netCDF4

    source, grid = _square_grid_file
    attacked = tmp_path / "transposed.nc"

    def mutate(src, dst):
        dst.createDimension("south_north_stag", 4)
        # Same shape, wrong axes: the only difference is the tuple.
        out = dst.createVariable(
            "z_obs", "f4", ("level", "west_east", "south_north_stag"),
            fill_value=np.float32(-9.99e30))
        out.units = "dBZ"
        out.description = src.variables["z_obs"].description
        out.coordinates = "XLONG XLAT"
        out[:] = np.zeros((2, 3, 4), dtype=np.float32)
        return {"z_obs"}

    _rewrite(source, attacked, mutate=mutate)
    # The identity attribute survived the rewrite untouched.
    with netCDF4.Dataset(attacked, "r") as dataset:
        assert dataset.grid_identity_sha256 == grid.identity_sha256()

    with pytest.raises(RadarGridSchemaError,
                       match=r"z_obs is declared over"):
        read_radar_grid(attacked)
    # And still refused when the caller demands exactly this grid, which is
    # the call the DA adapter makes.
    with pytest.raises(RadarGridSchemaError, match="south_north_stag"):
        read_radar_grid(attacked,
                        expected_grid_identity=grid.identity_sha256())


def test_the_dimension_tuple_is_checked_for_order_type_and_units(tmp_path, _square_grid_file):
    source, _ = _square_grid_file

    def transpose_plane(src, dst):
        out = dst.createVariable("z_err", "f4",
                                 ("level", "west_east", "south_north"))
        out.units = "dBZ"
        out[:] = np.zeros((2, 3, 3), dtype=np.float32)
        return {"z_err"}

    _rewrite(source, tmp_path / "swapped.nc", mutate=transpose_plane)
    with pytest.raises(RadarGridSchemaError, match="Dimension .*names and order"):
        read_radar_grid(tmp_path / "swapped.nc")

    def wrong_units(src, dst):
        # The source file's own velocity dimensions: this case is about
        # UNITS, so it must not accidentally trip the dimension check.
        out = dst.createVariable("vr_err", "f4",
                                 src.variables["vr_err"].dimensions)
        # Variances, not standard deviations -- still positive, still
        # finite, and a different quantity.
        out.units = "m2 s-2"
        out[:] = np.zeros(src.variables["vr_err"].shape, dtype=np.float32)
        return {"vr_err"}

    _rewrite(source, tmp_path / "variance.nc", mutate=wrong_units)
    with pytest.raises(RadarGridSchemaError, match="standard deviations"):
        read_radar_grid(tmp_path / "variance.nc")

    def wrong_dtype(src, dst):
        out = dst.createVariable("z_mask", "i4",
                                 ("level", "south_north", "west_east"))
        out[:] = np.zeros((2, 3, 3), dtype=np.int32)
        return {"z_mask"}

    _rewrite(source, tmp_path / "wide-mask.nc", mutate=wrong_dtype)
    with pytest.raises(RadarGridSchemaError, match="z_mask is stored as"):
        read_radar_grid(tmp_path / "wide-mask.nc")


def test_the_identity_is_bound_to_the_coordinates_the_file_stores(tmp_path, _square_grid_file):
    source, grid = _square_grid_file

    def move_a_gridpoint(src, dst):
        out = dst.createVariable("XLAT", "f4",
                                 ("south_north", "west_east"))
        out.units = "degree_north"
        out.description = src.variables["XLAT"].description
        values = np.asarray(src.variables["XLAT"][:], dtype=np.float32)
        values[0, 0] += np.float32(0.5)
        out[:] = values
        return {"XLAT"}

    _rewrite(source, tmp_path / "moved.nc", mutate=move_a_gridpoint)
    with pytest.raises(GridMismatchError, match="not the ones the grid identity"):
        read_radar_grid(tmp_path / "moved.nc")

    def drop_the_digests(src, dst):
        dst.grid_coordinate_sha256 = ""
        return set()

    _rewrite(source, tmp_path / "unbound.nc", mutate=drop_the_digests)
    with pytest.raises(RadarGridSchemaError, match="unbound claim"):
        read_radar_grid(tmp_path / "unbound.nc")


def test_a_scaled_beam_vector_is_refused_under_a_true_mask(tmp_path, _square_grid_file):
    source, _ = _square_grid_file

    def scale_the_beam(src, dst):
        # As above: this case is about beam MAGNITUDE, so it follows the
        # source's velocity layout rather than pinning one schema's.
        out = dst.createVariable(
            "vr_beam_east", "f4",
            src.variables["vr_beam_east"].dimensions,
            fill_value=np.float32(-9.99e30))
        out.units = "1"
        out[:] = np.asarray(src.variables["vr_beam_east"][:],
                            dtype=np.float32) * np.float32(2.0)
        return {"vr_beam_east"}

    _rewrite(source, tmp_path / "scaled.nc", mutate=scale_the_beam)
    # "longer than one", not "not unit length": the stored vector is the
    # MEAN of the contributing unit look directions, whose norm is the
    # cell's beam coherence and is 1 only when those beams were parallel.
    # The doubling is still caught, and by the invariant the mean actually
    # has rather than by one the merge stopped honouring.
    with pytest.raises(RadarGridSchemaError, match="longer than one"):
        read_radar_grid(tmp_path / "scaled.nc")


def test_grid_identity_moves_with_the_arrays_not_just_the_namelist():
    grid = _grid()
    same = _grid()
    assert grid.identity_sha256() == same.identity_sha256()

    # Same projection, same dims, one terrain point moved: a different grid.
    perturbed_terrain = grid.terrain_m.copy()
    perturbed_terrain[0, 0] += 1.0
    perturbed = TargetGrid(
        **{**{field: getattr(grid, field) for field in
              ("name", "map_proj", "nx", "ny", "nz", "dx_m", "dy_m",
               "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
               "lat", "lon", "z_w", "projection", "source")},
           "terrain_m": perturbed_terrain})
    assert perturbed.identity_sha256() != grid.identity_sha256()
    with pytest.raises(GridMismatchError, match="refusing to grid"):
        perturbed.require_identity(grid.identity_sha256())


def test_reflectivity_reduces_to_linear_mean_and_in_cell_maximum():
    """Two gates of 20 and 40 dBZ in one cell: mean is 37, not 30."""

    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    # A gate spacing far below the cell size guarantees co-location.
    volume = _volume(grid, reflectivity=[20.0, 40.0], velocity=None,
                     gate_size=250.0, first_gate=2125.0)
    observations, _ = _gridded(grid, volume)
    filled = observations.z_count > 0
    assert int(filled.sum()) == 1
    assert int(observations.z_count[filled][0]) == 2
    assert float(observations.z_max[filled][0]) == pytest.approx(40.0, abs=1e-4)
    expected_mean = 10.0 * np.log10((10 ** 2.0 + 10 ** 4.0) / 2.0)
    assert float(observations.z_mean[filled][0]) == pytest.approx(
        expected_mean, abs=1e-4)
    assert float(observations.z_mean[filled][0]) == pytest.approx(
        37.0, abs=0.05)
    # z_obs follows z_reduce.
    assert float(observations.z_obs[filled][0]) == pytest.approx(40.0,
                                                                 abs=1e-4)
    mean_reduced, _ = _gridded(grid, volume, z_reduce="mean")
    assert float(mean_reduced.z_obs[filled][0]) == pytest.approx(
        expected_mean, abs=1e-4)


def test_velocity_beyond_the_nyquist_fraction_is_masked_and_counted():
    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    params = SuperobParams(nyquist_reject_fraction=0.8, dealias=None)
    # 30 m/s against a 32 m/s Nyquist is 0.94 of it: aliasing risk.
    volume = _volume(grid, reflectivity=None, velocity=[30.0, 30.0],
                     nyquist=32.0)
    observations, _ = _gridded(grid, volume, params=params)
    assert int(observations.vr_mask.sum()) == 0
    assert int(observations.vr_rejected.sum()) == 2
    assert observations.counts[0]["velocity_gates_rejected_nyquist"] == 2

    # The same speeds against a 64 m/s Nyquist are fine.
    volume = _volume(grid, reflectivity=None, velocity=[30.0, 30.0],
                     nyquist=64.0)
    observations, _ = _gridded(grid, volume, params=params)
    assert int(observations.vr_mask.sum()) == 1
    assert float(observations.vr_obs[observations.vr_mask == 1][0]) == \
        pytest.approx(30.0, abs=1e-4)
    assert int(observations.vr_rejected.sum()) == 0


def test_a_sweep_without_nyquist_metadata_loses_every_velocity():
    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    volume = _volume(grid, reflectivity=[30.0, 30.0], velocity=[5.0, 5.0],
                     nyquist=None)
    observations, _ = _gridded(grid, volume)
    assert int(observations.vr_mask.sum()) == 0
    assert int(observations.vr_rejected.sum()) == 2
    assert observations.counts[0]["velocity_gates_rejected_no_nyquist"] == 2
    assert observations.counts[0]["sweeps_without_nyquist"] == 1
    # Reflectivity from the same sweep survives: the mask is per-moment.
    assert int(observations.z_mask.sum()) == 1


def test_an_implausible_nyquist_is_disbelieved_and_masks_its_whole_sweep():
    """A Nyquist velocity no weather radar has is metadata, not a threshold.

    The vendored parser read this field at the wrong byte until
    2026-07-30 and reported 620 m/s; the offset is fixed and pinned by a
    test in that crate, but a superob must not depend on the *next*
    parser being right either.  Anything outside the plausible band is
    treated exactly like a missing Nyquist: every velocity in the sweep
    is dropped and counted.
    """

    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    params = SuperobParams(nyquist_min_ms=4.0, nyquist_max_ms=100.0,
                           dealias=None)
    for absurd in (620.72, 0.5):
        volume = _volume(grid, reflectivity=[30.0, 30.0],
                         velocity=[5.0, 5.0], nyquist=absurd)
        observations, _ = _gridded(grid, volume, params=params)
        assert int(observations.vr_mask.sum()) == 0, absurd
        assert int(observations.vr_rejected.sum()) == 2, absurd
        assert observations.counts[0][
            "sweeps_with_implausible_nyquist"] == 1, absurd
        assert observations.counts[0]["sweeps_without_nyquist"] == 0, absurd
        # Reflectivity from the same sweep is untouched.
        assert int(observations.z_mask.sum()) == 1, absurd

    # A believable one at the same speeds is kept.
    volume = _volume(grid, reflectivity=None, velocity=[5.0, 5.0],
                     nyquist=23.84)
    observations, _ = _gridded(grid, volume, params=params)
    assert int(observations.vr_mask.sum()) == 1
    assert observations.counts[0]["sweeps_with_implausible_nyquist"] == 0


def test_a_cell_whose_velocities_span_the_nyquist_interval_is_dropped_whole():
    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    # shear_fold_fraction is lifted to 1.0 here so the *cell spread* rule is
    # what is under test.  At the 0.75 default these two gates are range
    # adjacent and 56 m/s apart, so the gate-to-gate scan reaches them
    # first; that path is tested separately below.
    params = SuperobParams(nyquist_reject_fraction=0.95,
                           nyquist_spread_fraction=0.5,
                           shear_fold_fraction=1.0, dealias=None)
    # +28 and -28 in one cell against Nyquist 32: a fold, not a shear.
    volume = _volume(grid, reflectivity=None, velocity=[28.0, -28.0],
                     nyquist=32.0)
    observations, _ = _gridded(grid, volume, params=params)
    assert int(observations.vr_mask.sum()) == 0
    assert observations.counts[0]["velocity_cells_rejected_spread"] == 1
    assert int(observations.vr_rejected.sum()) == 2


def test_the_gate_to_gate_scan_drops_the_gates_flanking_a_fold_boundary():
    """RAD-H3's achievable half: the *edge* of a folded region.

    At Nyquist 32 the interval is 64 m/s, so a single fold between
    neighbours shows up as a jump of nearly that.  +25 beside -25 is 50 m/s
    over one gate: above the 48 m/s threshold, and reachable only by a pair
    straddling the Nyquist limits in opposite directions -- which is the
    fold signature.  Both magnitudes are inside the 0.8 gate, so nothing
    else here would have touched them.
    """

    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    params = SuperobParams(nyquist_reject_fraction=0.9, dealias=None)
    volume = _volume(grid, reflectivity=None, velocity=[25.0, -25.0],
                     nyquist=32.0)
    observations, _ = _gridded(grid, volume, params=params)
    counts = observations.counts[0]
    assert counts["velocity_gates_rejected_nyquist"] == 0, "magnitude is fine"
    assert counts["velocity_gates_rejected_shear"] == 2
    assert counts["velocity_fold_boundaries"] == 1
    assert counts["velocity_gate_pairs_tested"] == 1
    assert counts["velocity_radials_fold_suspect"] == 1
    assert counts["velocity_sweeps_fold_suspect"] == 1
    assert int(observations.vr_mask.sum()) == 0

    # A real, strong, *unfolded* gradient of the same sign is not a
    # boundary: 5 -> 18 m/s is 13 m/s per gate, far above ordinary shear
    # and far below the Nyquist interval, and it survives untouched.
    volume = _volume(grid, reflectivity=None, velocity=[5.0, 18.0],
                     nyquist=32.0)
    observations, _ = _gridded(grid, volume, params=params)
    counts = observations.counts[0]
    assert counts["velocity_fold_boundaries"] == 0
    assert counts["velocity_gates_rejected_shear"] == 0
    assert int(observations.vr_mask.sum()) == 1


def test_a_coherent_fold_survives_every_mask_and_the_file_says_so(tmp_path):
    """The accurate half of RAD-H3, pinned so nobody re-reads the masks as proof.

    At Nyquist 32 a true +69 m/s folds coherently to +5 m/s.  A patch of
    gates that all fold together has a present and plausible Nyquist, speeds
    far inside the 0.8 threshold, zero in-cell spread, and no gate-to-gate
    jump anywhere in its interior.  It is assimilated, and this test exists
    to state that in the repository rather than in a comment.
    """

    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    # The masking-only path, STATED: this test is the record that four
    # risk masks do not exclude a coherent fold, which is why dealiasing
    # is now what a bare SuperobParams does.
    params = SuperobParams(dealias=None)
    volume = _volume(grid, reflectivity=None, velocity=[5.0, 5.0, 5.0, 5.0],
                     nyquist=32.0)
    observations, _ = _gridded(grid, volume, params=params)
    counts = observations.counts[0]
    assert int(observations.vr_mask.sum()) == 1
    assert float(observations.vr_obs[observations.vr_mask == 1][0]) == \
        pytest.approx(5.0, abs=1e-4)
    assert counts["velocity_gates_rejected_nyquist"] == 0
    assert counts["velocity_gates_rejected_shear"] == 0
    assert counts["velocity_cells_rejected_spread"] == 0
    assert counts["velocity_fold_boundaries"] == 0
    # ...and three gate pairs *were* examined, so the zero above is a
    # measurement rather than a scan that never ran.
    assert counts["velocity_gate_pairs_tested"] == 3

    # The file's own dealiasing attribute must not read as a clean bill.
    path = tmp_path / "coherent.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params)
    import netCDF4
    with netCDF4.Dataset(path, "r") as dataset:
        statement = dataset.dealiasing
    assert statement.startswith("none.")
    assert "signatures of risk, not" in statement
    assert "spatially coherent fold" in statement
    assert "true dealiasing" in statement

    read = read_radar_grid(path)
    record = read["provenance"]["fold_suspicion"][0][0]
    assert record["gate_pairs_tested"] == 3
    assert record["fold_boundaries"] == 0
    assert record["nyquist_ms"] == pytest.approx(32.0)
    assert read["provenance"]["dealiasing"] == statement


def test_the_beam_unit_vector_ships_with_the_velocity():
    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    # Due east, near-level: the look vector is almost exactly +east.
    volume = _volume(grid, reflectivity=None, velocity=[10.0],
                     azimuths=(90.0,), elevation=0.5, nyquist=64.0)
    observations, _ = _gridded(grid, volume)
    cell = observations.vr_mask == 1
    assert int(cell.sum()) == 1
    assert float(observations.vr_beam_east[cell][0]) == pytest.approx(1.0,
                                                                      abs=1e-3)
    assert abs(float(observations.vr_beam_north[cell][0])) < 1e-3
    assert 0.0 < float(observations.vr_beam_up[cell][0]) < 0.05
    norm = (observations.vr_beam_east[cell][0] ** 2
            + observations.vr_beam_north[cell][0] ** 2
            + observations.vr_beam_up[cell][0] ** 2)
    assert float(norm) == pytest.approx(1.0, abs=1e-9)


def test_two_radars_keep_separate_velocities_and_one_merged_reflectivity(
        tmp_path):
    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    # The masking-only path, STATED: this test is the record that four
    # risk masks do not exclude a coherent fold, which is why dealiasing
    # is now what a bare SuperobParams does.
    params = SuperobParams(dealias=None)
    first = superob_volume(
        _volume(grid, reflectivity=[30.0], velocity=[10.0], azimuths=(90.0,),
                site_id="KTLX", nyquist=64.0), grid, params=params)
    second = superob_volume(
        _volume(grid, reflectivity=[50.0], velocity=[-10.0], azimuths=(90.0,),
                site_id="KINX", nyquist=64.0), grid, params=params)
    observations = merge_contributions([first, second], grid, params=params)

    assert observations.vr_obs.shape[0] == 2
    assert observations.z_obs.shape == (grid.nz, grid.ny, grid.nx)
    filled = observations.z_count > 0
    assert int(observations.z_count[filled][0]) == 2
    assert float(observations.z_max[filled][0]) == pytest.approx(50.0,
                                                                 abs=1e-4)
    # The two velocities never average: each keeps its own sign.
    velocities = observations.vr_obs[:, filled]
    assert float(velocities[0, 0]) == pytest.approx(10.0, abs=1e-4)
    assert float(velocities[1, 0]) == pytest.approx(-10.0, abs=1e-4)

    path = tmp_path / "two-radars.nc"
    receipt = write_radar_grid(path, observations, grid,
                               valid_time="2026-07-28T20:03:16Z",
                               params=params)
    assert receipt["radars"] == ["KTLX", "KINX"]
    read = read_radar_grid(path)
    assert [radar["id"] for radar in read["radars"]] == ["KTLX", "KINX"]
    assert read["dims"]["radar"] == 2


def test_observation_error_is_floored_and_grows_with_in_cell_spread():
    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    params = SuperobParams(z_error_base_dbz=5.0, z_error_floor_dbz=2.0,
                           dealias=None)
    tight, _ = _gridded(grid, _volume(grid, reflectivity=[30.0] * 8,
                                      velocity=None), params=params)
    spread, _ = _gridded(grid, _volume(
        grid, reflectivity=[10.0, 50.0] * 4, velocity=None), params=params)
    cell = tight.z_count > 0
    assert float(tight.z_err[cell][0]) == pytest.approx(
        params.z_error_floor_dbz, abs=1e-9)
    assert float(spread.z_err[cell][0]) > 15.0


# -- SuperobParams validation ----------------------------------------------
#
# Every field below multiplies or bounds a physical threshold, so a value
# that cannot mean what the field is named for does not produce an obvious
# failure: it produces observations that are finite, plausible and wrong.
# Each field is refused at more than one bad value, and at both ends of its
# range where it has two.

#: ``(field, values)``, fractions of a Nyquist velocity or interval.
_BAD_FRACTIONS = [
    (name, [-0.1, -1.0, 1.01, 2.0, float("nan"), float("inf"),
            float("-inf")])
    for name in ("nyquist_reject_fraction", "nyquist_spread_fraction",
                 "shear_fold_fraction")
]

#: ``(field, values)``, quantities that are meaningless at or below zero.
#: For the four error fields, zero is the dangerous one: a zero standard
#: deviation is an infinitely confident observation, not a small error.
_BAD_POSITIVES = [
    (name, [0.0, -1.0, -1e-9, float("nan"), float("inf")])
    for name in ("nyquist_min_ms", "nyquist_max_ms", "max_range_km",
                 "z_error_base_dbz", "vr_error_base_ms",
                 "z_error_floor_dbz", "vr_error_floor_ms",
                 "refraction_factor", "earth_radius_m")
]


@pytest.mark.parametrize("field,values", _BAD_FRACTIONS + _BAD_POSITIVES)
def test_superob_params_refuse_each_field_at_several_bad_values(field, values):
    from woof.obs.superob import SuperobParamsError

    for value in values:
        with pytest.raises(SuperobParamsError, match=field):
            SuperobParams(**{field: value})


#: Runtime types that are not real numbers.  ``"250"`` and ``True`` are the
#: two the re-verification probe used; the rest are the same class of
#: mistake arriving from a config reader or a JSON payload.
_MALFORMED_TYPES = ["250", b"250", True, False, None, [250.0], (250.0,),
                    {"value": 250.0}, complex(250.0, 0.0)]


@pytest.mark.parametrize("value", _MALFORMED_TYPES)
@pytest.mark.parametrize("field", ["max_range_km", "nyquist_reject_fraction",
                                   "min_reflectivity_dbz", "earth_radius_m"])
def test_superob_params_refuse_a_runtime_type_that_is_not_a_number(field,
                                                                   value):
    """RAD-L1's residual gap: the validator did not validate its schema.

    ``SuperobParams(max_range_km="250", nyquist_reject_fraction=True)``
    constructed successfully, because ``validate`` read every field as
    ``float(getattr(self, name))`` -- ``float("250")`` is 250.0 and
    ``float(True)`` is 1.0, both in range -- and then kept the ORIGINAL
    object.  The string raised ``TypeError`` at the first arithmetic use
    (``params.max_range_km * 1000.0``), which is loud but is not the
    parameter validator doing its job, and the Rust side refuses these
    outright.

    Four fields across all three range families and nine types, because
    the fault is in how every field is read and not in any one of them.
    """

    from woof.obs.superob import SuperobParamsError

    with pytest.raises(SuperobParamsError, match=field):
        SuperobParams(**{field: value})


def test_superob_params_wrap_huge_integer_overflow_in_the_domain_error():
    """RV4-05: a Real too large for ``float`` is still a params refusal."""

    from woof.obs.superob import SuperobParamsError

    with pytest.raises(
            SuperobParamsError,
            match="max_range_km.*cannot be represented as a Python float"):
        SuperobParams(max_range_km=10 ** 1000)


def test_superob_params_normalize_the_real_numbers_they_accept():
    """Rejecting the malformed must not reject the merely un-normalized.

    An ``int`` and a NumPy scalar are real numbers and mean exactly what
    the field is named for; they are stored as Python floats so that
    ``to_payload`` and every arithmetic use see one runtime type.
    """

    params = SuperobParams(max_range_km=250, nyquist_min_ms=np.float32(4.0),
                           z_error_base_dbz=np.float64(5.0),
                           # The payload's float-only claim is about the
                           # numeric fields; the dealias block is a
                           # parameter object and is stated away here.
                           dealias=None)
    assert type(params.max_range_km) is float
    assert type(params.nyquist_min_ms) is float
    assert type(params.z_error_base_dbz) is float
    assert params.max_range_km == 250.0
    assert all(isinstance(value, float)
               for value in params.to_payload().values())


def test_a_malformed_type_smuggled_past_the_frozen_guard_is_refused(tmp_path):
    """And at the entry points, not only at construction.

    ``validate`` is called from every consumer precisely because
    construction is not the only way a parameter set arrives, so the type
    check has to live there too -- a set rebuilt from a JSON payload by a
    future reader would otherwise carry its strings straight into the
    pass.
    """

    from woof.obs.superob import SuperobParamsError

    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    volume = _volume(grid, reflectivity=[30.0] * 8, velocity=[5.0] * 8)
    good = SuperobParams()
    contribution = superob_volume(volume, grid, params=good)
    observations = merge_contributions([contribution], grid, params=good)

    for field, value in (("max_range_km", "250"),
                         ("nyquist_reject_fraction", True)):
        smuggled = SuperobParams()
        object.__setattr__(smuggled, field, value)
        with pytest.raises(SuperobParamsError, match=field):
            superob_volume(volume, grid, params=smuggled)
        with pytest.raises(SuperobParamsError, match=field):
            merge_contributions([contribution], grid, params=smuggled)
        with pytest.raises(SuperobParamsError, match=field):
            write_radar_grid(tmp_path / f"{field}.nc", observations, grid,
                             valid_time="2026-07-28T20:03:16Z",
                             params=smuggled)
        assert not (tmp_path / f"{field}.nc").exists()
    # The same field at its default is accepted, so the parametrization is
    # rejecting the value rather than the field name.
    assert SuperobParams(**{field: getattr(SuperobParams(), field)})


def test_superob_params_refuse_a_nyquist_band_that_is_not_a_band():
    from woof.obs.superob import SuperobParamsError

    # Inverted, and collapsed to a single point: both believe no reported
    # Nyquist at all, so every velocity in every sweep would be masked.
    for low, high in ((100.0, 4.0), (32.0, 32.0), (50.0, 49.999)):
        with pytest.raises(SuperobParamsError, match="not below"):
            SuperobParams(nyquist_min_ms=low, nyquist_max_ms=high)
    assert SuperobParams(nyquist_min_ms=4.0, nyquist_max_ms=100.0)


def test_superob_params_refuse_an_elevation_ceiling_that_is_not_an_angle():
    from woof.obs.superob import SuperobParamsError

    for value in (0.0, -1.0, 90.001, 180.0, 1e6, float("nan"),
                  float("inf")):
        with pytest.raises(SuperobParamsError, match="max_elevation_deg"):
            SuperobParams(max_elevation_deg=value)
    # The boundary itself is admissible: a vertical beam is an elevation.
    assert SuperobParams(max_elevation_deg=90.0)
    assert SuperobParams(max_elevation_deg=0.5)


def test_superob_params_refuse_a_reflectivity_floor_that_is_not_a_number():
    from woof.obs.superob import SuperobParamsError

    # A NaN floor compares False against every gate and empties the volume
    # in silence; an infinite one is not a floor either.  Any finite value,
    # including a positive one, is a legitimate choice.
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(SuperobParamsError, match="min_reflectivity_dbz"):
            SuperobParams(min_reflectivity_dbz=value)
    for value in (-35.0, 0.0, 20.0):
        assert SuperobParams(min_reflectivity_dbz=value)


def test_superob_params_are_revalidated_at_every_point_of_use(tmp_path):
    """Construction is not the only way an invalid parameter set arrives.

    ``frozen=True`` is a convention, not a wall: ``object.__setattr__``
    walks straight past it, and a set rebuilt from a stored payload by some
    future reader never runs ``__init__`` the same way.  The pass therefore
    checks what it is about to use, not what it was handed.
    """

    from woof.obs.superob import SuperobParamsError

    grid = _grid(nx=5, ny=5, dx=100000.0, nz=4, top_m=8000.0)
    volume = _volume(grid, reflectivity=[30.0] * 8, velocity=[5.0] * 8)

    # A sane set first, so the failures below are the mutation and not the
    # fixture.
    good = SuperobParams()
    contribution = superob_volume(volume, grid, params=good)
    observations = merge_contributions([contribution], grid, params=good)
    receipt = write_radar_grid(tmp_path / "good.nc", observations, grid,
                              valid_time="2026-07-28T20:03:16Z", params=good)
    assert receipt["schema"] == RADAR_GRID_SCHEMA

    # Now push a bad value past the frozen guard, two different ways, and
    # take it to each of the three entry points.
    for field, value in (("nyquist_reject_fraction", 4.0),
                         ("vr_error_base_ms", 0.0),
                         ("max_range_km", -250.0)):
        smuggled = SuperobParams()
        object.__setattr__(smuggled, field, value)
        with pytest.raises(SuperobParamsError, match=field):
            superob_volume(volume, grid, params=smuggled)
        with pytest.raises(SuperobParamsError, match=field):
            merge_contributions([contribution], grid, params=smuggled)
        with pytest.raises(SuperobParamsError, match=field):
            write_radar_grid(tmp_path / f"{field}.nc", observations, grid,
                             valid_time="2026-07-28T20:03:16Z",
                             params=smuggled)
        assert not (tmp_path / f"{field}.nc").exists(), (
            "a file must not be published with parameters that could not "
            "have produced it")


# ---------------------------------------------------------------------------
# when each radar's volume was scanned, and when it could be had
# ---------------------------------------------------------------------------


def test_per_radar_collection_instants_round_trip(tmp_path):
    """The file carries each radar's volume span and availability.

    ``valid_time`` per radar is the header start; the first and last
    radial instants and the feed's publication time ride beside it, so a
    reader can tell how far a radar's gates sit from the analysis time
    instead of dating every sweep to the header.
    """

    from dataclasses import replace

    grid = _grid()
    volume = replace(
        _volume(grid, reflectivity=np.linspace(5.0, 55.0, 40),
                velocity=np.linspace(-20.0, 20.0, 40)),
        start_time="2026-07-28T20:03:16.232Z",
        end_time="2026-07-28T20:09:56.232Z",
        key_time="2026-07-28T20:03:16Z", complete=True,
        sweeps_in_volume=12, sweeps_incomplete=0)
    params = SuperobParams(dealias=None)
    contribution = superob_volume(volume, grid, params=params)
    assert contribution.start_time == "2026-07-28T20:03:16.232Z"
    assert contribution.end_time == "2026-07-28T20:09:56.232Z"
    # The acquisition stage alone knows when the feed published it.
    contribution.availability_time = "2026-07-28T20:10:41.000Z"
    observations = merge_contributions([contribution], grid, params=params)
    assert observations.radars[0]["start_time"] == "2026-07-28T20:03:16.232Z"
    assert observations.radars[0]["availability_time"] \
        == "2026-07-28T20:10:41.000Z"
    assert observations.provenance[0]["end_time"] \
        == "2026-07-28T20:09:56.232Z"
    assert observations.provenance[0]["sweeps_in_volume"] == 12

    path = tmp_path / "radar-grid.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:15:00Z", params=params)
    read = read_radar_grid(path, expected_grid_identity=grid.identity_sha256())
    radar = read["radars"][0]
    assert radar["valid_time"] == "2026-07-28T20:03:16Z"
    assert radar["start_time"] == "2026-07-28T20:03:16.232Z"
    assert radar["end_time"] == "2026-07-28T20:09:56.232Z"
    assert radar["availability_time"] == "2026-07-28T20:10:41.000Z"
    assert read["valid_time"] == "2026-07-28T20:15:00Z"


def test_a_volume_that_did_not_say_when_it_was_scanned_reads_as_unknown(
        tmp_path):
    """Blank rows read as None, never as the start time standing in."""

    grid = _grid()
    volume = _volume(grid, reflectivity=np.linspace(5.0, 55.0, 40),
                     velocity=np.linspace(-20.0, 20.0, 40))
    assert volume.start_time is None and volume.end_time is None
    observations, params = _gridded(grid, volume)
    assert observations.radars[0]["end_time"] is None
    path = tmp_path / "radar-grid.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params)
    read = read_radar_grid(path, expected_grid_identity=grid.identity_sha256())
    radar = read["radars"][0]
    assert radar["valid_time"] == "2026-07-28T20:03:16Z"
    assert radar["start_time"] is None
    assert radar["end_time"] is None
    assert radar["availability_time"] is None
