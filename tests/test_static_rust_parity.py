"""Parity harness: the Rust static-field builder against the Python
implementation it ports.

RED BY DESIGN on the skeleton.  These tests are the acceptance floor
for the three port lanes (docs/dev/static-rust-port.md); each lane
turns its own class green and none may weaken an assertion to get
there.  Contract summary:

- **Public float64 projection**: exact platform NumPy comparisons.
- **WPS sampling**: portable Rust authority with independent NumPy bounds
  and cross-machine field hashes, owned by the portable golden tests.
  Full builds remain compared field by field against the NumPy reference
  under measured bounds, including categorical fields and source coverage.
- **Highres warp (lane 3)**: defined-behaviour tolerance parity for the
  warped planes; byte parity for everything downstream of them
  (triangle, crosswalk, donor fill, merges) and identical refusal
  decisions.

Dataset gating: builds run against the WPS_GEOG tree named by
``GPUWM_STATIC_PARITY_GEOG`` (default: the reference bundle path used
by the existing static suites).  Missing data SKIPS the build tests;
a missing or unimplemented BRIDGE FAILS them -- the whole point is
that the bare default must be the Rust path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.static_rust_parity

#: The reference WPS_GEOG tree (30-arc-second inventory).  Overridable
#: so the harness runs at the data on any box; the default is composed
#: from this account's home rather than written out, because a spelled
#: literal is one developer's absolute path and the release snapshot
#: refuses to ship a file carrying one.
GEOG_ROOT = Path(os.environ.get("GPUWM_STATIC_PARITY_GEOG")
                 or (Path.home() / "Downloads"
                     / "WRF_1974_MP55_reference_bundle" / "static"
                     / "WPS_GEOG"))

#: One mid-latitude Lambert parent + one sub-kilometre nest: the nest
#: exercises the float32 twin ULP paths (adopt_public_pole, compiler
#: bands), which is where stencil selection can silently diverge.
PARENT_SPEC = dict(
    kind="lambert", ref_lat=39.5, ref_lon=-84.0,
    truelat1=38.0, truelat2=41.0, stand_lon=-84.0,
    dx=9000.0, dy=9000.0, e_we=52, e_sn=45,
    known_x=26.0, known_y=22.5,
    moad_cen_lat=39.5, moad_cen_lon=-84.0,
)
NEST = dict(i_parent_start=18, j_parent_start=15, parent_grid_ratio=3,
            e_we=46, e_sn=40)


def _python_parent_grid():
    from woof.static.lambert import LambertGrid
    spec = PARENT_SPEC
    return LambertGrid(
        spec["ref_lat"], spec["ref_lon"], spec["truelat1"],
        spec["truelat2"], spec["stand_lon"], spec["dx"], spec["dy"],
        spec["e_we"], spec["e_sn"],
        known_x=spec["known_x"], known_y=spec["known_y"])


def _bridge_or_fail():
    """The bridge must load; anything else is a harness FAILURE.

    A skip here would let 'the Rust path is absent' read as green,
    which is the exact silent degradation fixed-means-default bans.
    """
    from woof.static import rust_bridge
    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        pytest.fail(
            "the Rust static-fields bridge is not loadable -- the "
            f"default static path cannot run: {reason}")
    return rust_bridge


class TestLane1GridParity:
    """Projection transforms and derived fields, byte-identical.

    Since the integration wave, bare array calls route to Rust by
    default -- so the oracle side here runs under the explicit
    WOOF_STATIC_PYTHON=1 fallback, keeping this a Rust-vs-numpy
    comparison rather than a self-comparison."""

    @pytest.mark.static_platform_qualification
    def test_mass_latlon_bytes_equal(self, monkeypatch):
        bridge = _bridge_or_fail()
        from woof.static import rust_bridge
        grid = _python_parent_grid()
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
        lat_py, lon_py = grid.latlon_mass()
        monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV)

        handle = bridge.grid_new(PARENT_SPEC)
        try:
            library = bridge.load()
            ny, nx = lat_py.shape
            for kind, expect in ((bridge.ARRAY_LAT, lat_py),
                                 (bridge.ARRAY_LON, lon_py)):
                out = np.empty((ny, nx), dtype=np.float64)
                code = library.gpuwm_static_grid_array(
                    handle, bridge.STAGGER_MASS, kind,
                    out.ctypes.data_as(
                        __import__("ctypes").POINTER(
                            __import__("ctypes").c_double)),
                    out.size)
                if code != 0:
                    pytest.fail("gpuwm_static_grid_array refused: "
                                + bridge.last_error(library))
                np.testing.assert_array_equal(
                    out, expect,
                    err_msg="Rust staggered lat/lon differs from the "
                            "Python transcription at the byte")
        finally:
            bridge.grid_free(handle)

    @pytest.mark.static_platform_qualification
    def test_nest_transform_roundtrip_bytes_equal(self, monkeypatch):
        bridge = _bridge_or_fail()
        from woof.static import rust_bridge
        parent_py = _python_parent_grid()
        child_py = parent_py.nest(
            NEST["i_parent_start"], NEST["j_parent_start"],
            NEST["parent_grid_ratio"], NEST["e_we"], NEST["e_sn"])
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
        lat_py, lon_py = child_py.latlon_mass()
        monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV)

        import ctypes
        library = bridge.load()
        parent = bridge.grid_new(PARENT_SPEC)
        child = ctypes.c_uint64(0)
        try:
            code = library.gpuwm_static_grid_nest(
                parent, NEST["i_parent_start"], NEST["j_parent_start"],
                NEST["parent_grid_ratio"], NEST["e_we"], NEST["e_sn"],
                float("nan"), float("nan"), ctypes.byref(child))
            if code != 0:
                pytest.fail("gpuwm_static_grid_nest refused: "
                            + bridge.last_error(library))
            out_lat = np.empty(lat_py.shape, dtype=np.float64)
            code = library.gpuwm_static_grid_array(
                child.value, bridge.STAGGER_MASS, bridge.ARRAY_LAT,
                out_lat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_lat.size)
            if code != 0:
                pytest.fail("child grid_array refused: "
                            + bridge.last_error(library))
            np.testing.assert_array_equal(out_lat, lat_py)
            del lon_py
        finally:
            bridge.grid_free(parent)
            if child.value:
                bridge.grid_free(int(child.value))


FULL_BUILD_BOUNDS = (Path(__file__).resolve().parents[1] / "tools" / "rustwx"
                    / "crates" / "static-fields" / "tests" / "goldens"
                    / "wps32" / "full-build-numpy-bounds.json")


def _assert_full_build_bounds(observed, expected, bounds):
    assert sorted(observed) == sorted(expected) == sorted(bounds), (
        "full-build field inventory differs")
    for name, cap in bounds.items():
        actual, reference = np.asarray(observed[name]), np.asarray(expected[name])
        assert actual.shape == reference.shape == tuple(cap["shape"]), name
        assert actual.dtype == reference.dtype == np.float64, name
        assert np.isfinite(actual).all() and np.isfinite(reference).all(), name
        delta = np.abs(actual - reference)
        assert float(delta.max()) <= cap["max_abs"], (
            f"{name}: full-build maximum difference exceeds the measured bound")
        assert float(delta.mean()) <= cap["mean_abs"], (
            f"{name}: full-build mean difference exceeds the measured bound")


@pytest.mark.parametrize("name", ["LANDMASK", "HGT_M", "LAI12M"])
def test_full_build_bounds_refuse_changed_kernel_output(name):
    """The reference gate must reject category, terrain and monthly drift."""
    cap = json.loads(FULL_BUILD_BOUNDS.read_text(encoding="utf-8"))["fields"][name]
    expected = {name: np.zeros(cap["shape"], dtype=np.float64)}
    observed = {name: expected[name].copy()}
    observed[name].flat[0] = cap["max_abs"] + 1.0
    with pytest.raises(AssertionError, match=name):
        _assert_full_build_bounds(observed, expected, {name: cap})


class TestLane2BuildParity:
    """Full independent builds, with measured per-field sampling bounds."""

    @pytest.fixture()
    def geog_root(self):
        if not (GEOG_ROOT / "topo_gmted2010_30s" / "index").is_file():
            pytest.skip(
                f"WPS_GEOG reference tree not present at {GEOG_ROOT} "
                "(set GPUWM_STATIC_PARITY_GEOG)")
        return GEOG_ROOT

    @pytest.mark.static_platform_qualification
    def test_build_static_matches_numpy_with_measured_bounds(self, geog_root, monkeypatch):
        _bridge_or_fail()
        from woof.static import rust_bridge
        from woof.static.build import GeogSelection, build_static

        table = json.loads(FULL_BUILD_BOUNDS.read_text(encoding="utf-8"))
        assert table["grid_spec"] == PARENT_SPEC
        # The measured fixture predates the permissive default soil set.
        # Compare the same named reference inputs; never apply its bounds
        # to a different climatology or include reference data in releases.
        from dataclasses import replace
        selection = replace(GeogSelection.fallback(geog_root),
                            resolution_tokens=("qualification-reference",),
                            **table["geography"])
        grid = _python_parent_grid()
        observed_coverage, expected_coverage = {}, {}
        monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV, raising=False)
        observed = build_static(grid, geog_root, selection=selection,
                                source_coverage_report=observed_coverage)
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
        expected = build_static(grid, geog_root, selection=selection,
                                source_coverage_report=expected_coverage)
        _assert_full_build_bounds(observed, expected, table["fields"])
        assert observed_coverage == expected_coverage, (
            "full-build source coverage differs from the independent reference")


#: The lane-3 golden fixtures (real Copernicus DEM windows + the
#: recorded tolerance caps) committed with the crate.
HIGHRES_FIXTURES = (Path(__file__).resolve().parent.parent
                    / "tools" / "rustwx" / "crates" / "static-fields"
                    / "tests" / "fixtures" / "highres")


def _highres_merge_inputs():
    """Baseline/override pair with spatially-varying climatologies so
    the donor CHOICE (not just the fill mechanism) is byte-compared."""
    shape = (12, 10)
    j = np.arange(shape[0])[:, None].astype(np.float64)
    i = np.arange(shape[1])[None, :].astype(np.float64)
    old_land = np.ones(shape)
    old_land[0:3, 0:3] = 0.0
    old_land[8, 7] = 0.0
    baseline = {
        "HGT_M": 300.0 + 5.0 * j + 3.0 * i,
        "LANDMASK": old_land,
        "LU_INDEX": np.where(old_land > 0.5, 12.0, 21.0),
        "LANDUSEF": np.zeros((21, *shape)),
        "SOILCTOP": np.zeros((16, *shape)),
        "SCT_DOM": np.full(shape, 6.0),
        "SOILCBOT": np.zeros((16, *shape)),
        "SCB_DOM": np.full(shape, 6.0),
        "GREENFRAC": 0.30 + 0.02 * j[None] + 0.001 * i[None]
        + 0.01 * np.arange(12)[:, None, None],
        "LAI12M": 1.0 + 0.1 * j[None] + 0.02 * i[None]
        + 0.05 * np.arange(12)[:, None, None],
        "ALBEDO12M": 14.0 + 0.05 * j[None] + 0.02 * i[None]
        + 0.2 * np.arange(12)[:, None, None],
        "SNOALB": 0.4 + 0.01 * j + 0.002 * i,
        "SOILTEMP": 278.0 + 0.5 * j + 0.25 * i,
        "TMN": 276.0 + 0.5 * j + 0.25 * i,
    }
    baseline["LANDUSEF"][11] = 1.0
    baseline["SOILCTOP"][5] = 1.0
    baseline["SOILCBOT"][5] = 1.0
    new_land = old_land.copy()
    new_land[0, 0] = 1.0
    new_land[1, 1] = 1.0
    new_land[11, 9] = 0.0
    new_land[5, 5] = 0.0
    overrides = {
        "HGT_M": baseline["HGT_M"] + 12.5,
        "LANDMASK": new_land,
        "LU_INDEX": np.where(new_land > 0.5, 12.0, 21.0),
        "LANDUSEF": baseline["LANDUSEF"].copy(),
        "SOILCTOP": baseline["SOILCTOP"].copy(),
        "SCT_DOM": baseline["SCT_DOM"].copy(),
        "SOILCBOT": baseline["SOILCBOT"].copy(),
        "SCB_DOM": baseline["SCB_DOM"].copy(),
    }
    return baseline, overrides


class TestLane3HighresParity:
    """Overlay compute: byte parity downstream of the warp, tolerance
    parity for the warp itself, identical refusal decisions.  Every
    Rust result here comes through the REAL cdylib (the routed default
    path); the Python reference runs under the reported
    WOOF_STATIC_PYTHON=1 fallback in the same process."""

    def test_highres_seam_entry_points_exist(self):
        bridge = _bridge_or_fail()
        library = bridge.load()
        for symbol in ("gpuwm_static_highres_terrain",
                       "gpuwm_static_highres_overrides",
                       "gpuwm_static_highres_merge",
                       "gpuwm_static_highres_derive_window",
                       "gpuwm_static_highres_fieldset_new",
                       "gpuwm_static_highres_audit_json",
                       "gpuwm_static_highres_usda"):
            assert hasattr(library, symbol), (
                f"highres seam entry point {symbol} missing")

    def test_default_path_is_rust_not_fallback(self):
        _bridge_or_fail()
        from woof.static.highres import _static_rust
        assert _static_rust("parity-probe") is not None, (
            "the bare default must route to the Rust seam")

    def test_usda_triangle_byte_parity_through_the_dll(self, monkeypatch):
        _bridge_or_fail()
        from woof.static import rust_bridge
        from woof.static.highres import usda_texture_category

        compositions = [(s, si, 100 - s - si)
                        for s in range(0, 101, 2)
                        for si in range(0, 101 - s, 2)]
        arr = np.asarray(compositions, dtype=np.float64)
        sand, silt, clay = (arr[:, 0][None, :] * 8.13,
                            arr[:, 1][None, :] * 8.13,
                            arr[:, 2][None, :] * 8.13)
        rust = usda_texture_category(sand, silt, clay)
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
        python = usda_texture_category(sand, silt, clay)
        np.testing.assert_array_equal(rust, python)

        with pytest.raises(ValueError) as rust_error:
            monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV)
            usda_texture_category(np.asarray([0.0]), np.asarray([0.0]),
                                  np.asarray([0.0]))
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
        with pytest.raises(ValueError) as python_error:
            usda_texture_category(np.asarray([0.0]), np.asarray([0.0]),
                                  np.asarray([0.0]))
        assert str(rust_error.value) == str(python_error.value)

    def test_merges_byte_parity_and_donor_choice_through_the_dll(
            self, monkeypatch):
        _bridge_or_fail()
        from woof.static import rust_bridge
        from woof.static.highres import (merge_highres_overrides,
                                          merge_terrain_override)

        baseline, overrides = _highres_merge_inputs()
        rust_merged, rust_audit = merge_highres_overrides(
            baseline, overrides)
        rust_terrain, rust_terrain_audit = merge_terrain_override(
            baseline, {"HGT_M": baseline["HGT_M"] + 41.5})
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
        python_merged, python_audit = merge_highres_overrides(
            baseline, overrides)
        python_terrain, python_terrain_audit = merge_terrain_override(
            baseline, {"HGT_M": baseline["HGT_M"] + 41.5})

        assert sorted(rust_merged) == sorted(python_merged)
        for name in sorted(python_merged):
            np.testing.assert_array_equal(
                np.asarray(rust_merged[name]),
                np.asarray(python_merged[name]),
                err_msg=f"{name}: Rust merge differs from the Python "
                        "reference at the byte")
        assert rust_audit == python_audit
        for name in sorted(python_terrain):
            np.testing.assert_array_equal(
                np.asarray(rust_terrain[name]),
                np.asarray(python_terrain[name]),
                err_msg=f"{name}: terrain-only merge differs")
        assert rust_terrain_audit == python_terrain_audit

    def test_merge_refusal_decisions_match_through_the_dll(
            self, monkeypatch):
        _bridge_or_fail()
        from woof.static import rust_bridge
        from woof.static.highres import merge_highres_overrides

        baseline, overrides = _highres_merge_inputs()
        baseline["SOILTEMP"] = np.zeros_like(baseline["SOILTEMP"])
        with pytest.raises(ValueError) as rust_error:
            merge_highres_overrides(baseline, overrides)
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
        with pytest.raises(ValueError) as python_error:
            merge_highres_overrides(baseline, overrides)
        assert str(rust_error.value) == str(python_error.value)
        assert "not a temperature" in str(rust_error.value)

    def test_warp_tolerance_on_the_pinned_real_footprint(self):
        """The seam terrain call (smoothing off, isolating the warp)
        against the Python rasterio path on the committed real
        Copernicus DEM window, gated by the caps recorded with the
        goldens."""
        bridge = _bridge_or_fail()
        clip = HIGHRES_FIXTURES / "terrain_clip.tif"
        meta_path = HIGHRES_FIXTURES / "meta.json"
        if not clip.is_file() or not meta_path.is_file():
            pytest.skip("lane-3 fixtures not present in this checkout")
        pytest.importorskip("rasterio")
        import ctypes
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        warp_meta = meta["terrain_warp"]
        spec = warp_meta["grid_spec"]
        halo = int(warp_meta["halo"])

        from woof.static.highres import (BoundRaster, _extended_grid,
                                          resample_continuous, sha256_file)
        from woof.static.lambert import LambertGrid

        grid = LambertGrid(
            spec["ref_lat"], spec["ref_lon"], spec["truelat1"],
            spec["truelat2"], spec["stand_lon"], spec["dx"], spec["dy"],
            spec["e_we"], spec["e_sn"], known_x=spec["known_x"],
            known_y=spec["known_y"])
        extended = _extended_grid(grid, halo)
        source = BoundRaster(
            path=clip, sha256=sha256_file(clip), source_id="parity",
            role="terrain", source_url="https://example.invalid/",
            license_id="test", license_url="https://example.invalid/",
            nominal_resolution="test")
        python_plane = resample_continuous(source, extended,
                                           method="average")

        library = bridge.load()
        request = json.dumps({
            "grid_spec": spec,
            "halo": halo,
            "smooth_passes": 0,
            "terrain": {"path": str(clip), "sha256": source.sha256},
        }).encode("utf-8")
        buffer = (ctypes.c_uint8 * len(request)).from_buffer_copy(request)
        handle = ctypes.c_uint64(0)
        code = library.gpuwm_static_highres_terrain(
            ctypes.c_uint64(0), buffer, len(request),
            ctypes.byref(handle))
        if code != 0:
            pytest.fail("gpuwm_static_highres_terrain refused: "
                        + bridge.last_error(library))
        try:
            fields = bridge.fieldset_to_dict(int(handle.value))
        finally:
            bridge.fieldset_free(int(handle.value))
        rust_plane = fields["HGT_M"]
        # smooth_passes=0 returns the CROPPED unsmoothed plane; crop
        # the Python extended plane the same way.
        crop = (slice(halo, halo + grid.e_sn - 1),
                slice(halo, halo + grid.e_we - 1))
        python_crop = python_plane[crop]
        assert rust_plane.shape == python_crop.shape
        delta = np.abs(rust_plane - python_crop)
        assert np.isfinite(rust_plane).all()
        assert delta.max() <= warp_meta["max_abs_delta_cap_m"], (
            f"max delta {delta.max():.2f} m beyond the recorded cap")
        assert delta.mean() <= warp_meta["mean_abs_delta_cap_m"], (
            f"mean delta {delta.mean():.3f} m beyond the recorded cap")

    @pytest.mark.parametrize("kind", ["global-terrain-window",
                                      "terrain-window"])
    def test_derive_window_writes_a_real_mosaic_through_the_dll(
            self, tmp_path, kind):
        """The terrain door's mosaic against rasterio.merge on the same
        lattice-snapped bounds (the fallback's arithmetic), asked for the
        footprint's own bounds as production asks.  The two clips share
        one pixel grid centred on the whole arc-seconds, so the declared
        1 arc-second lattice (latitude-banded sources) and the inherited
        one (staged tiles) are the same grid and meet the one golden."""
        bridge = _bridge_or_fail()
        west = HIGHRES_FIXTURES / "mosaic_west.tif"
        east = HIGHRES_FIXTURES / "mosaic_east.tif"
        meta_path = HIGHRES_FIXTURES / "meta.json"
        if not west.is_file() or not meta_path.is_file():
            pytest.skip("lane-3 fixtures not present in this checkout")
        rasterio = pytest.importorskip("rasterio")
        import ctypes
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mosaic_meta = meta["mosaic"]
        out_path = tmp_path / "derived_mosaic.tif"
        request = {
            "kind": kind,
            "tiles": [str(west), str(east)],
            "bounds": mosaic_meta["bounds_wsen"],
            "out_path": str(out_path),
        }
        if kind == "global-terrain-window":
            request.update(resolution_deg=mosaic_meta["resolution_deg"],
                           sea_level_fill=0.0)
        request = json.dumps(request).encode("utf-8")
        buffer = (ctypes.c_uint8 * len(request)).from_buffer_copy(request)
        cap = 65536
        out = (ctypes.c_uint8 * cap)()
        library = bridge.load()
        length = int(library.gpuwm_static_highres_derive_window(
            buffer, len(request), out, cap))
        if length < 0:
            pytest.fail("derive_window refused: "
                        + bridge.last_error(library))
        audit = json.loads(bytes(out[:length]).decode("utf-8"))
        assert audit["output_shape"] == mosaic_meta["shape"]
        assert audit["total_pixels"] == (mosaic_meta["shape"][0]
                                         * mosaic_meta["shape"][1])
        # rasterio can read the artifact the Rust writer produced, and it
        # sits on the lattice rasterio cut: same pixel grid, same place.
        with rasterio.open(out_path) as derived:
            values = derived.read(1)
            assert list(values.shape) == mosaic_meta["shape"]
            assert tuple(derived.transform)[:6] == pytest.approx(
                mosaic_meta["transform"], rel=0.0, abs=1e-12)
        expect = np.frombuffer(
            (HIGHRES_FIXTURES / "mosaic_filled.bin").read_bytes(),
            dtype=np.float32).reshape(values.shape)
        holes = np.frombuffer(
            (HIGHRES_FIXTURES / "mosaic_holes.bin").read_bytes(),
            dtype=np.uint8).reshape(values.shape).astype(bool)
        compared = ~holes
        if kind == "global-terrain-window":
            assert np.isfinite(values).all()   # holes take the sea-level fill
        assert np.isfinite(values[compared]).all()
        delta = np.abs(values[compared].astype(np.float64)
                       - expect[compared].astype(np.float64))
        assert delta.max() <= mosaic_meta["max_abs_delta_cap_m"]
        assert delta.mean() <= mosaic_meta["mean_abs_delta_cap_m"]


class TestEstateAndDoctor:
    """The finish contract: a user must be able to REACH the default.

    `build_static` runs on the crate by default, so the crate is now a
    prerequisite of the shipped configuration exactly as the NetCDF
    writer is: a wheel that cannot stage it is a wheel whose every
    static build is a reported workaround.  These bind the packaging
    estate and the doctor report to the seam's own constants.
    """

    def test_the_builder_is_a_bundled_artifact_with_one_spelling_everywhere(
            self):
        from woof import bridge_assets, bridges
        from woof.static import rust_bridge

        rows = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                if a.name == "static_fields"]
        assert rows, ("static_fields is not a bundled artifact: a wheel "
                      "install cannot reach the default static builder")
        artifact, = rows
        assert artifact.kind == "library"
        assert artifact.env_var == rust_bridge.STATIC_BRIDGE_ENV
        assert not artifact.vendored          # gpuwm-authored: stamp-proved
        assert (bridges.BRIDGE_ABI_MARKERS["static_fields"]
                == rust_bridge.ABI_MARKER)
        symbol, version = bridge_assets.library_abi_for("static_fields")
        assert symbol == "gpuwm_static_abi_version"
        assert version == rust_bridge.STATIC_ABI
        # The staged filename is the one the seam's own ladder opens.
        assert bridge_assets.artifact_filename(artifact, "win-x86_64") \
            == "static_fields.dll"
        assert bridge_assets.artifact_filename(artifact, "linux-x86_64") \
            == "libstatic_fields.so"
        from woof.bridges import default_bridge_dir
        staged = default_bridge_dir() / rust_bridge.library_names()[0]
        assert staged in set(rust_bridge.library_candidates())

    def test_doctor_reports_the_static_builder_row(self):
        from woof import doctor
        from woof.static import rust_bridge

        check = doctor._static_builder_check()
        assert "static builder" in check.name
        # The library is built and loadable in this tree (the parity
        # harness above fails otherwise), so the row must be verified
        # and carry the ABI evidence.
        assert rust_bridge.unavailable_reason() is None
        assert check.status == "verified"
        assert f"ABI {rust_bridge.STATIC_ABI}" in check.detail
        # The bundle-coverage sweep knows which line reports it.
        assert "static_fields" in doctor._CHECKED_ARTIFACTS
