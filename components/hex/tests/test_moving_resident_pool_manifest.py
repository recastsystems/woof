"""Resident moving pools carry their manifest in process; every reader takes it from there."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


def test_rrfs_orographic_check_reads_a_resident_projection_manifest(tmp_path):
    """THE BREAKAGE THIS PREVENTS (test setup G03, 2026-11-02): an hrrrv5 storm-following region move with
    resident pools failed with "pool.class0/manifest.json" missing: the projection keeps its manifest in this
    process (moving_pool_payload_v842.project writes only resident-payload.json), but the RRFS orographic-
    statistics check opened the file, so moving DA needed the disk fallback WOOF_HEX_MOVING_RESIDENT_POOLS=0."""
    from woof.hex.mpasphys import moving_pool_payload_v842 as resident
    from woof.hex.mpasphys.rrfs.framework_init_rrfs import FrameworkInitializationRRFS
    payload = tmp_path / "resident-lts" / "pool.class0"
    payload.mkdir(parents=True)
    (payload / "resident-payload.json").write_text("{}")
    fields = [dict(pool="sfc_input", name="var2d", origins=["ugwp_oro_data.nc"]),
              dict(pool="sfc_input", name="oa1", origins=["registry-initial-value"])]
    root = ET.fromstring("<registry><streams><stream name='ugwp_oro_data_in'>"
                         "<var name='var2d'/><var name='oa1'/></stream></streams></registry>")
    owner = SimpleNamespace(session=SimpleNamespace(layout=SimpleNamespace(root=root),
                                                    hydration=SimpleNamespace(payload=payload)))
    resident.clear()
    with resident._LOCK:
        resident._PAYLOADS[str(payload.resolve())] = SimpleNamespace(
            manifest=dict(schema="mpas-native-pool-payload-v1", fields=fields))
    try:
        # The check reads the resident manifest and still refuses a Registry initial value by name.
        with pytest.raises(RuntimeError, match="oa1"):
            FrameworkInitializationRRFS.check_orographic_statistics(owner)
        fields[1]["origins"] = ["ugwp_oro_data.nc"]
        FrameworkInitializationRRFS.check_orographic_statistics(owner)
    finally:
        resident.clear()


def test_payload_manifest_reads_the_file_without_a_resident_projection(tmp_path):
    import json
    from woof.hex.mpasphys.pool_v842 import payload_manifest
    (tmp_path / "manifest.json").write_text(json.dumps(dict(schema="mpas-native-pool-payload-v1", fields=[])))
    assert payload_manifest(tmp_path)["fields"] == []
