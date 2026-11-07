"""The reported large-global startup gates, without needing a large GPU host."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from woof.hex import run_owner
from woof.hex.device_admission import CardProfile, MIB
from woof.hex.host_lifetime import release_preparation
from woof.hex.mpasphys.memory_admission import rank_requirements


def test_dead_owner_reclaimed_live_unknown_and_foreign_preserved(tmp_path):
    path = tmp_path / 'scratch'
    code = "from woof.hex.run_owner import claim_directory; import sys; claim_directory(sys.argv[1]); print('ready',flush=True); sys.stdin.read()"
    child = subprocess.Popen([sys.executable, '-c', code, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        (path / 'sentinel').write_text('keep')
        with pytest.raises(FileExistsError):
            run_owner.claim_directory(path)
        assert (path / 'sentinel').read_text() == 'keep'
        child.kill()
        child.wait()
        run_owner.claim_directory(path)
        assert not (path / 'sentinel').exists()
        owner = json.loads((path / run_owner.MARKER).read_text())
        owner['host'] = 'another-host'
        (path / run_owner.MARKER).write_text(json.dumps(owner))
        with pytest.raises(FileExistsError):
            run_owner.claim_directory(path)
        (path / run_owner.MARKER).unlink()
        with pytest.raises(FileExistsError):
            run_owner.claim_directory(path)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_live_registered_rank_preserved_after_parent_death(tmp_path):
    path = tmp_path / 'samehost-ranks'
    run_owner.claim_directory(path)
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
    try:
        run_owner.register_child(path, child.pid)
        marker = path / run_owner.MARKER
        owner = json.loads(marker.read_text())
        # A different process start time models a dead, reused parent PID.
        owner['processes'][0]['started'] -= 1
        marker.write_text(json.dumps(owner))
        with pytest.raises(FileExistsError):
            run_owner.claim_directory(path)
        child.kill()
        child.wait()
        run_owner.claim_directory(path)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_mpas_10km_eight_96gb_cards_refused_and_shared_cards_summed():
    profile = CardProfile(name='96 GB card', multiprocessors=188, max_threads_per_sm=1536)
    cells = [-(5_898_242 // -8)] * 8  # lower bound: no halos yet
    required = rank_requirements(cells, range(8), dict.fromkeys(range(8), profile), 55)
    assert all(need > 97_887 * MIB for need in required.values())
    one = rank_requirements([40_962], [0], {0: profile}, 55)[0]
    # Two ranks on one card are co-tenants priced from the measured row (memory_admission.MPAS_COTENANT_RANK_ROWS),
    # not two whole members: never below one member of their combined columns, below two whole members.
    two = rank_requirements([40_962,40_962], [0,0], {0: profile}, 55)[0]
    assert one < two <= 2 * one
    assert rank_requirements([40_962], [0], {0: profile}, 80)[0] > one


def test_release_drops_global_arrays_and_keeps_receipt():
    values = np.ones((100, 100))
    reference = weakref.ref(values)
    host = dict(prepared=SimpleNamespace(mesh=values, expected_execution_fingerprint={'sha256':'ok'}),
                constructor_values={'v':values}, gwdo_host=values, config='keep')
    del values
    release_preparation(host)
    assert reference() is None
    assert host['prepared'].expected_execution_fingerprint == {'sha256':'ok'}
    assert host['config'] == 'keep'


def test_linked_tree_or_marker_never_authorizes_reclaim(tmp_path):
    target = tmp_path / 'target'
    run_owner.claim_directory(target)
    (target / 'sentinel').write_text('keep')
    marker = target / run_owner.MARKER
    owner = json.loads(marker.read_text())
    owner['processes'][0]['started'] -= 1
    marker.write_text(json.dumps(owner))
    linked = tmp_path / 'linked'
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        run_owner.claim_directory(linked)
    external = tmp_path / 'external-owner.json'
    marker.rename(external)
    marker.symlink_to(external)
    with pytest.raises(FileExistsError):
        run_owner.claim_directory(target)
    assert (target / 'sentinel').read_text() == 'keep'
    assert external.is_file()


def test_amd_trial_uses_input_bytes_and_sums_contexts_without_cuda_fit():
    from woof.hex.mpasphys.memory_admission import amd_capacity_trial
    required = amd_capacity_trial([120, 130], [0, 1], global_cells=200,
                                  input_payload_bytes=2000, reserve_bytes=100)
    assert required == {0: 1300, 1: 1400}
    assert amd_capacity_trial([120, 130], [0, 0], global_cells=200,
                             input_payload_bytes=2000, reserve_bytes=100) == {0: 2700}
    with pytest.raises(ValueError):
        amd_capacity_trial([0], [0], global_cells=200, input_payload_bytes=2000)


def test_named_native_rank_gate_precedes_single_context_price(monkeypatch):
    from woof.hex import forecast_door
    from woof.hex.mpasphys import memory_admission
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=False))))
    arguments = SimpleNamespace(samehost_devices='0,1')
    expected = {'devices': [0, 1], 'required_bytes': {0: 10, 1: 10}}
    monkeypatch.setattr(memory_admission, 'admit', lambda value: expected if value is arguments else None)
    assert forecast_door.admit_native_device(arguments, None) == expected
