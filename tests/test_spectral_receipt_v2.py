"""Pre-registration, hashing, gate, and input-identity controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from woof.verify import spectral
from woof.verify import spectral_io
from woof.verify import spectral_receipt

CURRENT_V1_PINS = "f3d1d17f80742c7114be8984cdfb55c10db117ecaa6e926b3ccc27414834062e"


def arrays(n: int = 64) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    x = np.arange(n, dtype=np.float64)
    base = np.cos(2.0 * np.pi * 4 * x / n)
    plane = np.tile(base, (n, 1))
    left = {
        "W": 2.0 * plane,
        "U": plane[np.newaxis, ...],
        "V": np.zeros((1, n, n), dtype=np.float64),
    }
    reference = {
        "W": plane,
        "U": plane[np.newaxis, ...],
        "V": np.zeros((1, n, n), dtype=np.float64),
    }
    return left, reference


def write_spec(path: Path, *, left: str, reference: str,
               with_gate: bool = True, missing_paths: bool = False) -> None:
    gate = """
[[gates]]
id = "w-power"
pair = "d01-f006"
field = "w-plane"
component = "scalar"
band = "mesoscale"
metric = "power_ratio"
minimum = 3.9
maximum = 4.1
""" if with_gate else ""
    path.write_text(f'''schema = "gpuwm.spectral-comparison-source/v1"
title = "synthetic WOOF-ArWen versus WRF"
left_label = "WOOF-ArWen"
reference_label = "WRF"
default_dx_m = 1000.0
crop_cells = 0

[[bands]]
name = "small"
minimum_km = 2.0
maximum_km = 8.0

[[bands]]
name = "mesoscale"
minimum_km = 8.0
maximum_km = 32.0

[[bands]]
name = "large"
minimum_km = 32.0

[[pairs]]
name = "d01-f006"
left_path = "{left}"
reference_path = "{reference}"
domain = "d01"
valid_time = "1974-04-03T18:00:00"

[[fields]]
name = "w-plane"
kind = "scalar"
variable = "W"
reduction = "plane"
units = "m s-1"

[[fields]]
name = "wind-level0"
kind = "vector"
u_variable = "U"
v_variable = "V"
reduction = "level"
level_index = 0
left_destagger = "none"
reference_destagger = "none"
{gate}''', encoding="utf-8")


def test_registration_does_not_open_or_require_model_output(tmp_path: Path):
    spec = tmp_path / "missing.toml"
    write_spec(spec, left="does-not-exist-left.npz",
               reference="does-not-exist-reference.npz", missing_paths=True)
    registration = spectral_receipt.make_registration(spec)
    assert registration["registration_sha256"]
    with pytest.raises(ValueError, match="not a readable file"):
        spectral_receipt.score_registration(registration)


def test_registration_hash_covers_bands_fields_gates_and_v1_identity(tmp_path: Path):
    spec = tmp_path / "spec.toml"
    write_spec(spec, left="left.npz", reference="reference.npz")
    registration = spectral_receipt.make_registration(spec)
    assert registration["parameters"]["legacy_spectral_v1_pins_sha256"] == (
        CURRENT_V1_PINS == spectral.PINS_SHA256 and CURRENT_V1_PINS)
    tampered = json.loads(json.dumps(registration))
    tampered["parameters"]["bands"][1]["minimum_km"] = 9.0
    with pytest.raises(ValueError, match="hash does not match"):
        spectral_receipt.validate_registration(tampered)


def test_receipt_scores_scalar_and_vector_and_passes_explicit_gate(tmp_path: Path):
    left, reference = arrays()
    np.savez(tmp_path / "left.npz", **left)
    np.savez(tmp_path / "reference.npz", **reference)
    spec = tmp_path / "spec.toml"
    write_spec(spec, left="left.npz", reference="reference.npz")
    registration = spectral_receipt.make_registration(spec)
    receipt = spectral_receipt.score_registration(registration)
    assert receipt["verdict"] == "pass"
    assert receipt["gates"]["passed"] == 1
    assert len(receipt["comparisons"]) == 2
    assert {item["kind"] for item in receipt["comparisons"]} == {
        "scalar", "vector"}
    assert len(receipt["inputs"]) == 2
    assert all(len(item["sha256"]) == 64 for item in receipt["inputs"])
    assert spectral_receipt.validate_receipt(receipt)["receipt_sha256"] == (
        receipt["receipt_sha256"])


def test_no_gate_is_explicitly_informational_not_a_green_gate(tmp_path: Path):
    left, reference = arrays()
    np.savez(tmp_path / "left.npz", **left)
    np.savez(tmp_path / "reference.npz", **reference)
    spec = tmp_path / "spec.toml"
    write_spec(spec, left="left.npz", reference="reference.npz",
               with_gate=False)
    receipt = spectral_receipt.score_registration(
        spectral_receipt.make_registration(spec))
    assert receipt["verdict"] == "informational"
    assert receipt["gates"]["rows"] == []


def test_a_gate_failure_is_not_hidden(tmp_path: Path):
    left, reference = arrays()
    np.savez(tmp_path / "left.npz", **left)
    np.savez(tmp_path / "reference.npz", **reference)
    spec = tmp_path / "spec.toml"
    write_spec(spec, left="left.npz", reference="reference.npz")
    registration = spectral_receipt.make_registration(spec)
    registration = json.loads(json.dumps(registration))
    registration["parameters"]["gates"][0]["maximum"] = 3.0
    registration["registration_sha256"] = spectral_receipt.canonical_hash(
        registration["parameters"])
    # A registration carries two digests: the raw one over the resolved
    # paths, and the portable policy one.  Re-registering by hand updates
    # both, or the validator refuses -- which is the point of the second.
    registration["registration_policy_sha256"] = (
        spectral_receipt.canonical_hash(
            spectral_receipt.policy_parameters(
                registration["parameters"])))
    receipt = spectral_receipt.score_registration(registration)
    assert receipt["verdict"] == "fail"
    assert receipt["gates"]["failed"] == 1


def test_an_unresolved_gate_is_incomplete_not_pass(tmp_path: Path):
    left, reference = arrays()
    np.savez(tmp_path / "left.npz", **left)
    np.savez(tmp_path / "reference.npz", **reference)
    spec = tmp_path / "spec.toml"
    write_spec(spec, left="left.npz", reference="reference.npz")
    registration = spectral_receipt.make_registration(spec)
    registration = json.loads(json.dumps(registration))
    gate = registration["parameters"]["gates"][0]
    gate["band"] = "large"
    gate["minimum_reference_power"] = 1e30
    registration["registration_sha256"] = spectral_receipt.canonical_hash(
        registration["parameters"])
    # A registration carries two digests: the raw one over the resolved
    # paths, and the portable policy one.  Re-registering by hand updates
    # both, or the validator refuses -- which is the point of the second.
    registration["registration_policy_sha256"] = (
        spectral_receipt.canonical_hash(
            spectral_receipt.policy_parameters(
                registration["parameters"])))
    receipt = spectral_receipt.score_registration(registration)
    assert receipt["verdict"] == "incomplete"
    assert receipt["gates"]["incomplete"] == 1


def test_receipt_tamper_and_input_tamper_are_detected(tmp_path: Path):
    left, reference = arrays()
    left_path = tmp_path / "left.npz"
    reference_path = tmp_path / "reference.npz"
    np.savez(left_path, **left)
    np.savez(reference_path, **reference)
    spec = tmp_path / "spec.toml"
    write_spec(spec, left="left.npz", reference="reference.npz")
    receipt = spectral_receipt.score_registration(
        spectral_receipt.make_registration(spec))
    tampered = json.loads(json.dumps(receipt))
    tampered["comparisons"][0]["result"]["bands"][1]["power_ratio"] = 1.0
    with pytest.raises(ValueError, match="receipt hash"):
        spectral_receipt.validate_receipt(tampered)
    np.savez(left_path, **{**left, "W": left["W"] + 1.0})
    with pytest.raises(ValueError, match="input moved"):
        spectral_receipt.validate_receipt(receipt, rehash_inputs=True)


def _load_calibrator():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "spectral_gate_calibrate.py"
    spec = importlib.util.spec_from_file_location("spectral_gate_calibrate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibrated_known_good_envelope_passes_member_and_rejects_outlier(
        tmp_path: Path):
    calibrator = _load_calibrator()
    receipts = []
    n = 64
    x = np.arange(n, dtype=np.float64)
    base = np.tile(np.cos(2.0 * np.pi * 4 * x / n), (n, 1))

    def score(scale: float, directory: Path) -> dict:
        directory.mkdir()
        wind_u = base[np.newaxis, ...]
        wind_v = np.zeros((1, n, n), dtype=np.float64)
        np.savez(directory / "left.npz", W=scale * base, U=wind_u, V=wind_v)
        np.savez(directory / "reference.npz", W=base, U=wind_u, V=wind_v)
        spec = directory / "spec.toml"
        write_spec(spec, left="left.npz", reference="reference.npz",
                   with_gate=False)
        return spectral_receipt.score_registration(
            spectral_receipt.make_registration(spec))

    for index, scale in enumerate((0.99, 1.01, 1.02), start=1):
        receipts.append(score(scale, tmp_path / f"member{index}"))
    gates, hashes = calibrator.calibrate(
        receipts, percentile=95.0,
        metrics=("power_ratio", "normalized_error_power"),
        minimum_samples=3)
    assert len(hashes) == 3
    assert spectral_receipt.evaluate_gates(
        receipts[-1]["comparisons"], gates)["verdict"] == "pass"
    outlier = score(1.10, tmp_path / "outlier")
    assert spectral_receipt.evaluate_gates(
        outlier["comparisons"], gates)["verdict"] == "fail"
    rendered = calibrator.render_toml(
        gates, receipt_hashes=hashes, percentile=95.0)
    assert all(digest in rendered for digest in hashes)
    assert "Never include the" in rendered


def test_plot_manifest_is_bound_to_the_receipt(tmp_path: Path):
    from woof.verify import spectral_plot

    left, reference = arrays()
    np.savez(tmp_path / "left.npz", **left)
    np.savez(tmp_path / "reference.npz", **reference)
    spec = tmp_path / "spec.toml"
    write_spec(spec, left="left.npz", reference="reference.npz",
               with_gate=False)
    receipt = spectral_receipt.score_registration(
        spectral_receipt.make_registration(spec))
    manifest = spectral_plot.plot_receipt(receipt, tmp_path / "plots")
    assert manifest["receipt_sha256"] == receipt["receipt_sha256"]
    assert len(manifest["files"]) == 16
    assert all(Path(row["path"]).is_file() for row in manifest["files"])
    assert (tmp_path / "plots" / "manifest.json").is_file()


def test_npz_refuses_a_nonzero_time_index_without_named_dimensions(
        tmp_path: Path):
    source = tmp_path / "field.npz"
    np.savez(source, W=np.zeros((8, 8), dtype=np.float64))
    with pytest.raises(ValueError, match="no named time dimension"):
        spectral_io.load_array(source, variable="W", time_index=1)


def test_a_history_file_with_no_suffix_is_read_as_netcdf(tmp_path: Path):
    """The reader opens the filenames the model actually writes.

    A forecast history file is named ``wrfout_d0N_<valid time>`` and carries
    no extension at all -- the shipped example spec in
    ``configs/verify/spectral_compare_example.toml`` points at exactly such a
    path.  A suffix allowlist therefore refused every real model output while
    every synthetic ``.npz`` control passed, so the whole NetCDF lane could
    read nothing a campaign would ever hand it.
    """

    netcdf4 = pytest.importorskip("netCDF4")
    source = tmp_path / "wrfout_d02_2020-01-01_00_00_00"
    with netcdf4.Dataset(source, "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("bottom_top", 3)
        dataset.createDimension("south_north", 4)
        dataset.createDimension("west_east_stag", 6)
        variable = dataset.createVariable(
            "U", "f4",
            ("Time", "bottom_top", "south_north", "west_east_stag"))
        variable.stagger = "X"
        variable.units = "m s-1"
        variable[0] = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)

    array, metadata = spectral_io.load_array(
        source, variable="U", time_index=0)
    assert array.shape == (3, 4, 6)
    assert array.dtype == np.float64
    assert metadata["format"] == "netcdf"
    assert metadata["netcdf_identified_by"] == "signature"
    assert metadata["stagger_attribute"] == "X"
    assert metadata["dimensions"] == [
        "bottom_top", "south_north", "west_east_stag"]


def test_a_suffixless_file_that_is_not_netcdf_is_refused_by_its_bytes(
        tmp_path: Path):
    """Widening the name rule must not widen what counts as model output.

    Accepting any extensionless path would hand GRIB messages, logs and
    truncated transfers to ``netCDF4`` and surface whatever it raised.  The
    signature is what admits a file, so a non-NetCDF byte stream is still a
    refusal, and the refusal says which test it failed.
    """

    source = tmp_path / "wrfout_d02_2020-01-01_00_00_00"
    source.write_bytes(b"GRIB" + bytes(64))
    with pytest.raises(ValueError, match="no NetCDF or HDF5 signature"):
        spectral_io.load_array(source, variable="T", time_index=0)

    empty = tmp_path / "wrfout_d02_2020-01-01_00_15_00"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="no NetCDF or HDF5 signature"):
        spectral_io.load_array(empty, variable="T", time_index=0)


def test_registration_refuses_invalid_side_overrides_and_fractional_level(
        tmp_path: Path):
    base = tmp_path / "base.toml"
    write_spec(base, left="left.npz", reference="reference.npz")
    text = base.read_text(encoding="utf-8")

    bad_destagger = tmp_path / "bad-destagger.toml"
    bad_destagger.write_text(
        text.replace('left_destagger = "none"',
                     'left_destagger = "diagonal"'),
        encoding="utf-8")
    with pytest.raises(ValueError, match="destagger must be one of"):
        spectral_receipt.make_registration(bad_destagger)

    bad_level = tmp_path / "bad-level.toml"
    bad_level.write_text(
        text.replace("level_index = 0", "level_index = 0.5"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="level_index must be an integer"):
        spectral_receipt.make_registration(bad_level)
