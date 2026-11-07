"""TC-VI1: vortex relocation and intensity correction on mapped (GEFS) preparations.

Breakage pinned: the Milton box M GEFS ensemble could not take --vortex-init (GFS-only refusal), so its first data-assimilation
analysis started from weak, broad member vortices. CPU-only checks; the GPU transform itself is exercised by a real
GEFS member preparation (the 2026-10-26 Milton observation fix).
"""
import argparse
import hashlib
import json
import math

import numpy as np
import pytest

from woof.tc import mapped_vortex as mv
from woof.tc.frame_transform import derived_mslp

# Milton operational TCVITALS, gfs.t18z.syndata.tcvitals.tm00 (rw_tc_vitals record shape)
RECORD = {"atcf_id": "AL142024", "valid_time": "2024-10-07T18:00:00Z", "latitude_deg": 21.7,
          "longitude_deg": -91.3, "minimum_pressure_pa": 90900.0, "maximum_wind_m_s": 77.0,
          "radius_maximum_wind_m": 9000.0}


def vitals(tmp_path, records=(RECORD,)):
    path = tmp_path / "vitals.json"
    path.write_text(json.dumps({"schema": "tc-vitals/1", "records": [{"vitals": r} for r in records]}))
    return path


def args(**kw):
    p = argparse.ArgumentParser()
    mv.add_arguments(p)
    return p.parse_args([x for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", str(v))])


def test_declaration_and_command(tmp_path):
    path = vitals(tmp_path)
    assert mv.declaration(args()) is None
    doc = mv.declaration(args(vortex_init="relocate", vortex_vitals=path))
    assert doc["mode"] == "relocate" and doc["vitals_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert mv.command_arguments(args(vortex_init="relocate", vortex_vitals=path)) == [
        "--vortex-init", "relocate", "--vortex-vitals", str(path.resolve())]
    with pytest.raises(ValueError):
        mv.declaration(args(vortex_init="relocate"))
    with pytest.raises(ValueError):
        mv.declaration(args(vortex_vitals=path))


def test_vitals_record_is_exact(tmp_path):
    later = dict(RECORD, valid_time="2024-10-08T00:00:00Z", longitude_deg=-90.4)
    path = vitals(tmp_path, (RECORD, later))
    assert mv.vitals_record(path, "2024-10-07T18:00:00")["longitude_deg"] == -91.3
    assert mv.vitals_record(path, "2024-10-08T00:00:00")["longitude_deg"] == -90.4
    with pytest.raises(ValueError):
        mv.vitals_record(path, "2024-10-07T21:00:00")   # intensity targets are never interpolated
    bad = vitals(tmp_path, (dict(RECORD, radius_maximum_wind_m=None),))
    with pytest.raises(ValueError):
        mv.vitals_record(bad, "2024-10-07T18:00:00")


def test_derived_mslp_is_surface_pressure_at_sea_level_and_reduces_over_land():
    psfc = np.array([95000.0, 90000.0])
    z = np.array([0.0, 1000.0])
    t2 = np.array([300.0, 290.0])
    q2 = np.array([0.02, 0.01])
    mslp = derived_mslp(psfc, z, t2, q2)
    assert mslp[0] == 95000.0
    tv = 290.0 * (1 + .6078 * .01) + .5 * .0065 * 1000.0
    assert math.isclose(mslp[1], 90000.0 * math.exp(9.80665 * 1000.0 / (287.04 * tv)), rel_tol=1e-12)
    assert 100000.0 < mslp[1] < 101500.0


def test_validate_evidence_chain(tmp_path):
    root = tmp_path / "prepared"
    evidence = {"declaration.json": b"{}\n", "vitals.json": b"{}\n", "member-diagnosis.json": b"{}\n",
                "background-frames.json": b"before\n", "relocated-frames.json": b"after\n",
                "library.sha256": b"x\n"}
    sha = {n: hashlib.sha256(b).hexdigest() for n, b in evidence.items()}
    evidence["tc-transform.json"] = json.dumps({
        "declaration": {"valid_time": "2024-10-07T18:00:00", "background_manifest_sha256": sha["background-frames.json"]},
        "output_manifest_sha256": sha["relocated-frames.json"]}).encode()
    (root / "source-evidence").mkdir(parents=True)
    mv.write_evidence(root / "source-evidence", evidence)
    binding = {"schema": mv.SCHEMA, "valid_time": "2024-10-07T18:00:00", "mode": "relocate",
               "evidence_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in evidence.items()}}
    mv.validate_evidence(root, binding, valid_time="2024-10-07T18:00:00")
    with pytest.raises(ValueError):
        mv.validate_evidence(root, binding, valid_time="2024-10-08T00:00:00")
    (root / "source-evidence/vortex-init/vitals.json").write_bytes(b"changed\n")
    with pytest.raises(ValueError):
        mv.validate_evidence(root, binding, valid_time="2024-10-07T18:00:00")


def test_source_cli_admits_mapped_sources(tmp_path, capsys):
    from woof import source_cli
    path = vitals(tmp_path)
    # the mapped door needs --vortex-vitals; the GFS-only refusal is retired for mapped sources
    rc = source_cli.main(["--source", "gefs", "--vortex-init", "relocate", "--dry-run"])
    assert rc != 0 and "--vortex-vitals" in capsys.readouterr().err
    rc = source_cli.main(["--source", "gfs", "--vortex-init", "relocate", "--vortex-vitals", str(path), "--dry-run"])
    assert rc != 0 and "--vortex-inputs" in capsys.readouterr().err
