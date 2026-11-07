"""Public forcing selection, integrity, and optional setup identity."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def source(tmp_path, monkeypatch):
    from woof import public_cumulus_aerosol as module
    data=b"public-forcing-test-transport"
    metadata={"output_bytes":len(data),"output_sha256":hashlib.sha256(data).hexdigest(),"source_grid_stride":2}
    pin=tmp_path/"source.json";pin.write_text(json.dumps(metadata))
    monkeypatch.setattr(module,"SOURCE",pin)
    monkeypatch.setenv("WOOF_PUBLIC_AEROSOL_CACHE",str(tmp_path/"cache"))
    monkeypatch.delenv("WOOF_PUBLIC_AEROSOL",raising=False)
    return module,data


def test_legacy_filename_is_never_selected(tmp_path,monkeypatch):
    module,data=source(tmp_path,monkeypatch)
    (tmp_path/"CESM_RCP4.5_Aerosol_Data.dat").write_bytes(data)
    class NoLegacy(Exception):pass
    def builder(*args):raise NoLegacy()
    monkeypatch.setattr(module,"binary",builder)
    with pytest.raises(NoLegacy):module.public_asset(tmp_path)
    public=tmp_path/module.FILENAME;public.write_bytes(data)
    asset,receipt=module.public_asset(tmp_path)
    assert asset==public and receipt["label"]==module.LABEL


def test_corruption_and_missing_override_fail_before_use(tmp_path,monkeypatch):
    module,data=source(tmp_path,monkeypatch)
    public=tmp_path/module.FILENAME;public.write_bytes(data)
    module.public_asset(tmp_path)
    public.write_bytes(b"X"+data[1:])
    with pytest.raises(ValueError,match="differs from the pinned"):module.public_asset(tmp_path)
    monkeypatch.setenv("WOOF_PUBLIC_AEROSOL",str(tmp_path/"absent"))
    with pytest.raises(FileNotFoundError):module.public_asset(tmp_path)


def test_option_off_never_reads_public_data(monkeypatch):
    from woof.core.cumulus_aerosol import restore_monthly_aerosol,update_monthly_aerosol
    state=SimpleNamespace()
    restore_monthly_aerosol(state,{},SimpleNamespace(aercu_opt=0),None)
    update_monthly_aerosol(state,SimpleNamespace(aercu_opt=0),None)
    assert vars(state)=={}


def test_receipt_is_setup_identity():
    from woof.io.restart import classify_state_attr
    from woof.state_serialization_contract import STATE_OPTIONAL_SETUP_JSON
    assert classify_state_attr("aerosol_forcing")=="setup"
    assert "aerosol_forcing" in STATE_OPTIONAL_SETUP_JSON
