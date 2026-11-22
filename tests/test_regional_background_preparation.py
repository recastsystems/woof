"""The actual input reader and local snapshot retain regional identities."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from woof.regional_preparation import read_prepared, review_background


@pytest.mark.parametrize('source', ['gfs', 'era5', '20crv3'])
def test_portable_reader_preserves_native_source_cache_and_member(tmp_path, monkeypatch, source):
    from test_prepared_single_domain_forecast import _prepared_fixture
    from woof import prepared_single_domain_forecast as reader
    from woof.experiment import load_experiment
    fixture = _prepared_fixture(tmp_path, source, physics_profile=None)
    # The shared fixture carries a minimal source/cache payload. Geography
    # interpolation is outside this identity and dispatch control.
    monkeypatch.setattr(reader, 'validate_native_lambert_contract',
                        lambda exp, path, *, source_name: SimpleNamespace(source=source))
    monkeypatch.setattr(reader, 'verify_native_static_receipt', lambda *args: {'status': 'PASS'})
    # HGT_M: since c1227a11a the reader verifies the terrain auto-smoothing
    # receipt against the static terrain; a flat plane carries none.
    monkeypatch.setattr(reader, 'load_native_static_cache',
                        lambda path, grid, ny, nx: {'STATIC': np.ones((ny, nx)),
                                                    'HGT_M': np.zeros((ny, nx))})
    exp = load_experiment(fixture.experiment)
    inputs = read_prepared(fixture.prepared, fixture.experiment, fixture.wps,
                          source=source, run_seconds=fixture.run_seconds,
                          history_interval_seconds=exp.root.history_interval_s)
    assert inputs.source == source
    assert inputs.prepared_cache_path.is_dir()
    assert inputs.forcing_hours[0] == 0
    assert inputs.source_member == ('072' if source == '20crv3' else None)
    before = fixture.experiment.read_bytes()
    fixture.experiment.write_bytes(before + b'\n# changed authority\n')
    with pytest.raises(ValueError, match='differ|SHA|sha|identity'):
        read_prepared(fixture.prepared, fixture.experiment, fixture.wps,
                      source=source, run_seconds=fixture.run_seconds,
                      history_interval_seconds=exp.root.history_interval_s)


@pytest.mark.parametrize('committed_state', [False, True])
def test_stale_owned_preparation_preserves_old_files_and_rebuilds_before_members(tmp_path, monkeypatch, committed_state):
    from test_prepared_single_domain_forecast import _prepared_fixture
    from woof import prepared_single_domain_forecast as reader
    from woof import regional_preparation as preparation
    from woof.background_contract import plan as select
    from woof.experiment import load_experiment
    from datetime import datetime, timezone
    fixture = _prepared_fixture(tmp_path, 'gfs', physics_profile=None)
    monkeypatch.setattr(reader, 'validate_native_lambert_contract',
                        lambda exp, path, *, source_name: SimpleNamespace(source=source_name))
    monkeypatch.setattr(reader, 'verify_native_static_receipt', lambda *args: {'status': 'PASS'})
    # HGT_M: since c1227a11a the reader verifies the terrain auto-smoothing
    # receipt against the static terrain; a flat plane carries none.
    monkeypatch.setattr(reader, 'load_native_static_cache',
                        lambda path, grid, ny, nx: {'STATIC': np.ones((ny, nx)),
                                                    'HGT_M': np.zeros((ny, nx))})
    exp = load_experiment(fixture.experiment)
    initial = exp.start_time.replace(tzinfo=timezone.utc)
    selected = select('gfs', init=initial, now=datetime(2026, 9, 12, tzinfo=timezone.utc),
                      run_seconds=fixture.run_seconds,
                      cadence_hours=json.loads(fixture.proof.read_text())['boundary_interval_seconds'] // 3600)
    background = dict(schema=preparation.SCHEMA, selection=selected.record(),
                      inputs={'kind': 'automatic', 'files': []}, fetch_hints=selected.fetch_hints(), supplements=[])
    plan = dict(background=background, review_sha256='c' * 64,
                selected={'cadence_seconds': exp.root.history_interval_s}, request={})
    (tmp_path / 'experiment.toml').write_bytes(fixture.experiment.read_bytes())
    (tmp_path / 'experiment.namelist.wps').write_bytes(fixture.wps.read_bytes())
    monkeypatch.setattr(preparation, 'validate_saved_background', lambda *args: selected)
    # A real v3 physics receipt from another registry reaches the ordinary
    # reader's compatibility refusal. No input-reader arithmetic is replaced.
    # The receipt names a registry document whose physics this build cannot
    # establish: since A153 a document digest alone (a citation edit) is
    # not a physics difference, so the receipt drops the physics parts that
    # would resolve it, as a receipt from an unknown registry carries none.
    proof = json.loads(fixture.proof.read_text())
    proof['physics'].pop('registry_physics')
    proof['physics']['registry_sha256'] = '0' * 64
    fixture.proof.write_text(json.dumps(proof))
    stale_bytes = fixture.proof.read_bytes()
    result = dict(prepared_root=str(fixture.prepared), experiment_config=str(fixture.experiment),
                  wps_namelist=str(fixture.wps))
    receipt = dict(review_sha256=plan['review_sha256'], background=background, result=result,
                   files=preparation._files([fixture.proof, fixture.source_manifest, fixture.experiment, fixture.wps]))
    record = tmp_path / 'background-preparation.json'
    record.write_text(json.dumps(receipt))
    original_receipt = record.read_bytes()
    calls = []
    def prepare(*args, run_dir, **kwargs):
        calls.append(run_dir)
        inner = run_dir / 'native'
        inner.mkdir()
        fresh = _prepared_fixture(inner, 'gfs', physics_profile=None)
        return dict(prepared_root=str(fresh.prepared), experiment_config=str(fresh.experiment),
                    wps_namelist=str(fresh.wps))
    monkeypatch.setattr(preparation, 'preparation_chains', lambda: {'prepared:go': prepare})
    if committed_state:
        states = tmp_path / 'cycles' / 'cycle_000'
        states.mkdir(parents=True)
        np.savez(states / 'analysis.npz', state=np.array([1.]))
        with pytest.raises(ValueError, match='committed regional state'):
            preparation.prepare_background(plan, tmp_path, exp, geog=tmp_path / 'geog')
        assert not calls
        assert record.read_bytes() == original_receipt
    else:
        inputs = preparation.prepare_background(plan, tmp_path, exp, geog=tmp_path / 'geog')
        assert len(calls) == 1
        assert inputs.prepared_root != fixture.prepared
        saved = json.loads(record.read_text())
        assert Path(saved['recovery']['previous_receipt']).read_bytes() == original_receipt
        assert saved['recovery']['previous_files_preserved'] is True
        preparation.prepare_background(plan, tmp_path, exp, geog=tmp_path / 'geog')
        assert len(calls) == 1, 'a current cache is restored without preparation'
    assert fixture.proof.read_bytes() == stale_bytes


@pytest.mark.parametrize('source,member', [('gefs', 'c00'), ('gefs', 'p30'), ('aigefs', 'mem000'), ('aigefs', 'mem030')])
def test_prepared_table_member_uses_the_bound_primary_inputs(tmp_path, monkeypatch, source, member):
    from test_forcing_member import fixture
    from woof import forcing_member
    from woof.regional_preparation import _prepared_member
    from woof.source_adapters import get_source_adapter
    hints, handoff, path, row = fixture(tmp_path, monkeypatch, source=source, member=member)
    receipt = forcing_member.verify_handoff(hints, handoff, out=tmp_path)
    root = tmp_path / 'prepared'
    root.mkdir()
    (root / 'forcing-member.json').write_text(json.dumps(receipt))
    inputs = SimpleNamespace(source_member=None, prepared_root=root,
                             source_manifest={'primary_files': [{'sha256': receipt['files'][0]['sha256']}]})
    request = SimpleNamespace(source_member=member)
    assert _prepared_member(request, inputs, get_source_adapter(source)) == member
    # The existing fixture injects only decoded record inventory. The real
    # grammar and its complete identity-field verifier remain active here.
    row['member'] = '999'
    with pytest.raises(ValueError, match='member'):
        _prepared_member(request, inputs, get_source_adapter(source))
    row['member'] = str(forcing_member.member_contract(source, member)[1].member(member).ordinal)
    inputs.source_manifest['primary_files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='primary input order'):
        _prepared_member(request, inputs, get_source_adapter(source))


@pytest.mark.parametrize('member', [0, 9])
def test_supplied_ensemble_product_reaches_its_native_member_owner(tmp_path, monkeypatch, member):
    from datetime import datetime
    from woof import era5_member
    from woof.regional_preparation import _prepared_member, _files
    from woof.source_adapters import get_source_adapter
    path = tmp_path / 'source.grib'
    path.write_bytes(b'role binding control; native inventory is substituted')
    inputs = SimpleNamespace(source_member=None, source_manifest={'files': {'grib': _files([path])[0]}},
                             boundary_interval_seconds=10800, experiment=SimpleNamespace(start_time=datetime(2026, 7, 29)))
    request = SimpleNamespace(source_product='ensemble_members', source_member=member,
                              forcing_cadence_hours=3, source_provider='cds', source_inputs={'grib': str(path)})
    calls = []
    monkeypatch.setattr(era5_member, 'check_member', lambda source, number: calls.append((source, number)))
    assert _prepared_member(request, inputs, get_source_adapter('era5')) == member
    assert calls == [(path, member)]
    request.source_product = None
    with pytest.raises(ValueError, match='reanalysis has no EDA member'):
        _prepared_member(request, inputs, get_source_adapter('era5'))
    request.source_product = 'ensemble_members'
    path.write_bytes(b'changed original bytes')
    with pytest.raises(ValueError, match='differ from the prepared source manifest'):
        _prepared_member(request, inputs, get_source_adapter('era5'))
    assert len(calls) == 1

