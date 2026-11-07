"""Byte preservation and failed-publication checks, not forecast evidence."""
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pytest

from woof.io.restart import _publish_restart_payload


@pytest.fixture
def tool():
    path = os.environ.get('TEST_RESTART_ZIP_TOOL')
    if not path:
        pytest.skip('set TEST_RESTART_ZIP_TOOL to the compiled compressor')
    return path


def test_parallel_archive_preserves_every_npy_byte(tmp_path, monkeypatch, tool):
    # Signed zero, NaN payloads and subnormals must survive without numerical
    # decoding. Include Fortran order and a nested array name used in restarts.
    words = np.array([0,0x80000000,0x7fc01234,1,0x3f800000], dtype=np.uint32)
    payload = {'header':np.frombuffer(b'{"clock":900}', dtype=np.uint8),
               'state/thp':words.view(np.float32),
               'physics/leaves':np.asfortranarray(np.arange(30).reshape(5,6))}
    plain = tmp_path/'plain.npz'
    compressed = tmp_path/'compressed.npz'
    monkeypatch.delenv('WOOF_RESTART_ZIP_TOOL',raising=False)
    _publish_restart_payload(plain,payload)
    monkeypatch.setenv('WOOF_RESTART_ZIP_TOOL',tool)
    monkeypatch.setenv('WOOF_RESTART_ZIP_THREADS','3')
    _publish_restart_payload(compressed,payload)
    with zipfile.ZipFile(plain) as a, zipfile.ZipFile(compressed) as b:
        assert a.namelist()==b.namelist()
        for name in a.namelist():
            assert a.read(name)==b.read(name)
            assert b.getinfo(name).compress_type==zipfile.ZIP_DEFLATED
    with np.load(compressed,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['state/thp'].view(np.uint32),words)


def test_failed_compression_cannot_replace_committed_checkpoint(tmp_path,monkeypatch,tool):
    target=tmp_path/'checkpoint.npz'
    target.write_bytes(b'previous complete generation')
    monkeypatch.setenv('WOOF_RESTART_ZIP_TOOL',tool)
    monkeypatch.setenv('WOOF_RESTART_ZIP_THREADS','0')
    with pytest.raises(ValueError,match='positive'):
        _publish_restart_payload(target,{'field':np.arange(8)})
    assert target.read_bytes()==b'previous complete generation'
    assert list(tmp_path.iterdir())==[target]


def test_tool_refuses_to_clobber_an_existing_archive(tmp_path,tool):
    source=tmp_path/'source.npz'
    np.savez(source,field=np.arange(8))
    destination=tmp_path/'destination'
    destination.write_bytes(b'owned by another writer')
    result=subprocess.run([tool,str(source),str(destination),'2'],capture_output=True)
    assert result.returncode
    assert destination.read_bytes()==b'owned by another writer'
