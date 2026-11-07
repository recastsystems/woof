from argparse import Namespace
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from woof.hex import preset_prepare as prep


def test_native_suite_preparation_keeps_its_own_soil_and_species():
    from woof.hex.physics_suites import load_table, suite_selectors
    from woof.hex.species_row import species_row_for_scheme

    selectors = suite_selectors("hrrrv5", "conus3km")
    table = load_table()
    assert selectors == table["mpas_suites"]["suites"]["hrrrv5"]
    soil = table["options"]["config_lsm_scheme"][selectors["config_lsm_scheme"]]
    assert soil["carried"]["n_soil_levels"] == 9
    assert selectors["config_microp_scheme"] == "mp_tempo"
    assert {"ng", "volg", "nwfa", "nifa"} <= set(
        species_row_for_scheme(selectors["config_microp_scheme"]).names())
    # The former preparation workaround selected a different microphysics row.
    assert suite_selectors("rrfs_v2_dev")["config_microp_scheme"] == "mp_thompson_aerosols"


def test_cycle_selects_packaged_public_source_and_vertical_contract():
    from woof.hex.source_profiles import load_source_profiles
    from importlib import resources
    row = next(row for row in load_source_profiles().values() if "mesh_set" in row)
    args = Namespace(source_profile=row["id"], mesh=next(iter(row["mesh_set"]["meshes"])),
                     cycle="2026100312", hours=6)
    cycle, model, mesh, source = prep.request(args)
    assert cycle.hour == 12 and mesh["cells"] > 0
    assert source["role"] == "preparation_source"
    spec = resources.files("woof.hex").joinpath(model["vertical"]["preparation_vertical_spec"])
    assert len(json.loads(spec.read_text())["specified_interfaces_m"]) == model["vertical"]["interfaces"]
    import hashlib
    table_row = source["intermediate"]["vtable"]
    table = resources.files("woof.hex").joinpath(table_row["resource"]).read_bytes()
    assert hashlib.sha256(table).hexdigest() == table_row["sha256"]
    # The native WPS reader needs ordinary meteorological names and numeric
    # GRIB1 columns, even for GRIB2. The adaptation table has neither.
    assert any(line.split("|")[0].strip().isdigit() and line.split("|")[4].strip() == "TT"
               for line in table.decode().splitlines() if "|" in line)
    args.cycle = "2026100313"
    with pytest.raises(ValueError, match="six-hour cycle"):
        prep.request(args)


@pytest.mark.parametrize("length,etag,valid", [(3, "pinned", True), (2, "pinned", False), (3, "changed", False)])
def test_public_object_identity_and_truncation(tmp_path, monkeypatch, length, etag, valid):
    import io
    @contextmanager
    def response(*args, **kwargs):
        stream = io.BytesIO(b"abc"[:length])
        stream.headers = {"ETag": '"' + etag + '"'}
        yield stream
    monkeypatch.setattr(prep, "urlopen", response)
    target = tmp_path / "asset.nc"
    row = {"bytes": 3, "s3_etag": "pinned"}
    if not valid:
        with pytest.raises(ValueError):
            prep.acquire("https://public.example/asset", row, target)
        assert not target.exists()
    else:
        record = prep.acquire("https://public.example/asset", row, target)
        assert prep.acquire("https://public.example/asset", row, target) == record
        target.write_bytes(b"abd")
        with pytest.raises(ValueError, match="matching source bytes"):
            prep.acquire("https://public.example/asset", row, target)


def test_failed_door_keeps_log_and_never_claims_success(tmp_path):
    import sys
    log = tmp_path / "failed.log"
    with pytest.raises(ValueError, match="exit 7"):
        prep.checked_command([sys.executable, "-c", "print('native defect'); raise SystemExit(7)"], log)
    assert "native defect" in log.read_text()
    assert json.loads(log.with_suffix(".timing.json").read_text())["returncode"] == 7
    prep.checked_command([sys.executable, "-c", "print('recovered')"], log)
    assert "native defect" in log.read_text()
    assert "recovered" in log.with_name("failed.attempt-1.log").read_text()


def test_retry_reuses_the_same_source_config_and_refuses_a_changed_one(tmp_path):
    path = tmp_path / "vertical.json"
    prep.publish_config(path, b"pinned")
    prep.publish_config(path, b"pinned")
    with pytest.raises(ValueError, match="mixed preparation"):
        prep.publish_config(path, b"changed")
    assert path.read_bytes() == b"pinned"


def test_preparation_assets_carry_what_the_forecast_door_reads_beside_the_grid():
    """Every asset name resolves per mesh, and the suite's drag statistics land where the door looks."""
    from woof.hex.source_profiles import load_source_profiles
    row = next(row for row in load_source_profiles().values() if "mesh_set" in row)
    assets = row["mesh_set"]["preparation_assets"]
    for mesh, table in row["mesh_set"]["meshes"].items():
        names = {name.format(mesh=mesh): key for name, key in assets.items()}
        # forecast_door: Path(grid).parent / f"{mesh}.ugwp_oro_data.nc"
        assert names[f"{mesh}.ugwp_oro_data"] == "ugwp_oro_data"
        assert {"grid", "static", "invariant"} <= set(names)
        for key in names.values():
            assert table[key]["bytes"] > 0 and table[key]["s3_etag"]
