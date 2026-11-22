"""Actual native section pixels for inherited WOOF model/source labels."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pytest

from woof import cli, render_layout
from test_render_rust import RENDERER, needs_renderer


def _write_section_fixture(directory):
    """The stored-plane fixture, written by the shipped Rust netcdf-writer.

    A line-for-line port of ``stored_plane_fixture::write`` (tools/rustwx/
    crates/rw-wrfbatch/tests/stored_plane_fixture/mod.rs): the same CDF-2
    header, attributes, variables and float32 values, PH/PHB included, so
    a section can be drawn.  It goes through ``libnetcdf_writer`` (the
    bridge every native build ships) rather than the crate's Cargo test
    binary, which no release build produces: the gate's tipbuild natives
    and CI's build-native step carry the bridge and not the test binary,
    so the test refused with "native section fixture writer not built" on
    the 2.8.7 Stage 1 gate.
    """
    from woof.io.nc_writer_bridge import ClassicSchema

    f4 = np.float32
    nx, ny, nz = 24, 18, 4
    cells = nx * ny
    valid_time = "2026-08-19_00:00:00"
    path = directory / f"wrfout_d01_{valid_time.replace(':', '_')}"
    schema = ClassicSchema("cdf2")
    time = schema.def_dim("Time", 0, unlimited=True)
    strlen = schema.def_dim("DateStrLen", 19)
    bottom_top = schema.def_dim("bottom_top", nz)
    bottom_top_stag = schema.def_dim("bottom_top_stag", nz + 1)
    south_north = schema.def_dim("south_north", ny)
    south_north_stag = schema.def_dim("south_north_stag", ny + 1)
    west_east = schema.def_dim("west_east", nx)
    west_east_stag = schema.def_dim("west_east_stag", nx + 1)
    for name, value in (("TITLE", " OUTPUT FROM WRF V4.6.1 MODEL"),
                        ("START_DATE", valid_time),
                        ("SIMULATION_START_DATE", valid_time),
                        ("GRIDTYPE", "C")):
        schema.put_global_attr(name, value)
    for name, value in (("MAP_PROJ", 1), ("GRID_ID", 1), ("PARENT_ID", 0)):
        schema.put_global_attr(name, value, np.int32)
    for name, value in (("DX", 3000.0), ("DY", 3000.0), ("TRUELAT1", 38.0),
                        ("TRUELAT2", 38.0), ("STAND_LON", -95.0),
                        ("CEN_LAT", 38.0), ("CEN_LON", -95.0),
                        ("POLE_LAT", 90.0), ("POLE_LON", 0.0)):
        schema.put_global_attr(name, value, np.float32)
    times = schema.def_var("Times", "S1", (time, strlen))

    def variable(name, dims, units, *, describe=False):
        varid = schema.def_var(name, "f4", dims)
        schema.put_var_attr(varid, "units", units)
        if describe:
            schema.put_var_attr(varid, "description", name)
        return varid

    surface_rows = (("XLAT", "degree_north"), ("XLONG", "degree_east"),
                    ("T2", "K"), ("TH2", "K"), ("Q2", "kg kg-1"),
                    ("PSFC", "Pa"), ("HGT", "m"), ("U10", "m s-1"),
                    ("V10", "m s-1"), ("SINALPHA", "1"), ("COSALPHA", "1"),
                    ("MU", "Pa"), ("MUB", "Pa"), ("TSK", "K"),
                    ("MSLP_ANOM", "Pa"))
    surface = {name: variable(name, (time, south_north, west_east), units,
                              describe=True)
               for name, units in surface_rows}
    mass_levels = (time, bottom_top, south_north, west_east)
    full_levels = (time, bottom_top_stag, south_north, west_east)
    volume = {}
    for name, dims, units in (
            ("P_TOP", (time,), "Pa"), ("P_HYD", mass_levels, "Pa"),
            ("T", mass_levels, "K"), ("P", mass_levels, "Pa"),
            ("PB", mass_levels, "Pa"), ("QVAPOR", mass_levels, "kg kg-1"),
            ("QCLOUD", mass_levels, "kg kg-1"),
            ("QICE", mass_levels, "kg kg-1"),
            ("ZNW", (time, bottom_top_stag), ""),
            ("PH", full_levels, "m2 s-2"), ("PHB", full_levels, "m2 s-2"),
            ("U", (time, bottom_top, south_north, west_east_stag), "m s-1"),
            ("V", (time, bottom_top, south_north_stag, west_east), "m s-1")):
        volume[name] = variable(name, dims, units)

    row, col = (index.astype(np.float32) for index in np.indices((ny, nx)))
    t2 = f4(295.0) + f4(0.1) * (row + col)
    terrain, dry_mass, theta = f4(320.0), f4(96_000.0), f4(310.0)
    eta = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
    top, cloud = f4(4_000.0), f4(1.0e-4)
    interface = top + (f4(1.0) + (f4(0.008) + cloud)) * dry_mass * eta
    hydrostatic = f4(0.5) * (interface[:-1] + interface[1:])
    kappa = f4(0.28589641)
    layer_t = [theta * (level * f4(1.0e-5)) ** kappa for level in hydrostatic]
    humidity = f4(0.008) / f4(1.008)
    vapor = f4(0.0)
    for level, temperature in zip(hydrostatic[::-1], layer_t[::-1]):
        vapor += (f4(9.81) * (level / (f4(287.04) * temperature))
                  * f4(1_000.0) * humidity)
    shelter = ((dry_mass + top + vapor)
               * np.exp(f4(-0.068283) / layer_t[0]))
    plane = lambda value: np.full((ny, nx), value, dtype=np.float32)
    column = lambda values: np.repeat(
        np.asarray(values, dtype=np.float32), cells).reshape(-1, ny, nx)
    surface_values = {
        "XLAT": f4(36.0) + f4(0.05) * row, "XLONG": f4(-98.0) + f4(0.05) * col,
        "T2": t2, "TH2": t2 / (shelter * f4(1.0e-5)) ** kappa,
        "Q2": plane(0.010), "PSFC": plane(97_000.0), "HGT": plane(terrain),
        "U10": plane(6.0), "V10": plane(-4.0), "SINALPHA": plane(0.0),
        "COSALPHA": plane(1.0), "MU": plane(0.0), "MUB": plane(dry_mass),
        "TSK": plane(290.0),
        "MSLP_ANOM": (row * f4(nx) + col) - f4(120.0)}
    volume_values = {
        "P_TOP": np.array([top], dtype=np.float32),
        "P_HYD": column(hydrostatic), "T": column([theta - f4(300.0)] * nz),
        "P": column([0.0] * nz),
        "PB": column([97_000.0 - 12_000.0 * level for level in range(nz)]),
        "QVAPOR": column([0.008] * nz), "QCLOUD": column([cloud] * nz),
        "QICE": column([0.0] * nz), "ZNW": eta,
        "PH": column([0.0] * (nz + 1)),
        "PHB": column([f4(9.81) * (terrain + f4(1_000.0) * f4(level))
                       for level in range(nz + 1)]),
        "U": np.full((nz, ny, nx + 1), 7.0, dtype=np.float32),
        "V": np.full((nz, ny + 1, nx), -3.0, dtype=np.float32)}
    with schema.create(path) as writer:
        writer.write_record(0, times, valid_time.encode("ascii"))
        for name, varid in surface.items():
            writer.write_record(0, varid, surface_values[name])
        for name, varid in volume.items():
            writer.write_record(0, varid, volume_values[name])
    return path


def _native_section_fixture(tmp_path):
    """A PH-backed section-compatible wrfout, or the file the caller names."""
    explicit = os.environ.get("WOOF_SECTION_TEST_FIXTURE")
    if explicit:
        source = Path(explicit)
        assert source.is_file(), source
        return source
    directory = tmp_path / "native-section-fixture"
    directory.mkdir()
    return _write_section_fixture(directory)


@needs_renderer
@pytest.mark.parametrize("layout", ["fixed", "auto"])
def test_inherited_section_version_labels_match_resolved_literals_and_keep_generic_pixels(
        tmp_path, layout):
    source = _native_section_fixture(tmp_path)
    version_theme = tmp_path / "version.json"
    literal_theme = tmp_path / "literal.json"
    version_theme.write_text(json.dumps({"extends": "woof-light", "text": {
        "source_label": "Recast WOOF {version}", "model_label": "WOOF {version}"}}), encoding="utf-8")
    literal_theme.write_text(json.dumps({"extends": "woof-light", "text": {
        "source_label": "Recast WOOF 2.8.6", "model_label": "WOOF 2.8.6"}}), encoding="utf-8")

    def render(name, *extra):
        out = tmp_path / name
        args = ["render", str(source), "--engine", "rust", "--products", "xsec:tk",
                "--timeidx", "0", "--out", str(out), "--source-label", "WOOF-ArWen 2.8.6",
                "--section", "36.2,-97.8,36.6,-97.2"]
        if layout == "fixed":
            args += ["--size", "640x480"]
        else:
            args += ["--size", "auto"]
        assert cli.main([*args, *extra]) == 0
        pictures = list(out.rglob("*.png"))
        assert len(pictures) == 1, pictures
        return Path(render_layout.fs_path(pictures[0])).read_bytes()

    generic = render("generic")
    default = render("default", "--theme", "default")
    versioned = render("versioned", "--theme", str(version_theme))
    literal = render("literal", "--theme", str(literal_theme))
    assert generic == default
    assert versioned == literal
    assert versioned != generic
    evidence = os.environ.get("WOOF_RENDERER_EXTRAS_EVIDENCE")
    if evidence:
        root = Path(evidence)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"section-{layout}-generic.png").write_bytes(generic)
        (root / f"section-{layout}-woof-version.png").write_bytes(versioned)
        (root / f"section-{layout}-receipt.json").write_text(json.dumps({
            "schema": "renderer-extras.section-theme-proof.v1", "fixture": "stored_plane_fixture d01 3km with PH/PHB, written by the Rust netcdf-writer bridge",
            "layout": layout, "generic_default_byte_identical": generic == default,
            "versioned_literal_byte_identical": versioned == literal,
            "generic_sha256": hashlib.sha256(generic).hexdigest(),
            "woof_sha256": hashlib.sha256(versioned).hexdigest(),
        }, indent=2) + "\n", encoding="utf-8")
