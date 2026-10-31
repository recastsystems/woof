"""A retained receipt is not proof that a disposable member artifact still exists."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from woof.hex import moving_members as m, init_sources


@pytest.mark.parametrize('physics', [[], ['--physics-suite=convection_permitting']])
@pytest.mark.parametrize('missing, expected', [
    (None, []), ('grid', ['mesh', 'static', 'init']),
    ('static', ['static', 'init']), ('init', ['init']),
])
def test_retired_artifact_rebuilds_it_and_dependents(tmp_path, monkeypatch, missing, expected, physics):
    name = 'target'
    case = {'mesh': {}, 'physics_args': physics}
    if missing is None and not physics:
        expected = ['init']  # the promoted recipe cannot reuse an unprepared cold init
    monkeypatch.setattr(m, 'load_case', lambda _: case)
    monkeypatch.setattr(m, 'read_stage', lambda _: {'source': 'gfs', 'cycle': '2025-01-01_00:00:00', 'met': 'met.nc'})
    monkeypatch.setattr(m, 'resolve_vertical_spec', lambda _: {})
    monkeypatch.setattr(m, 'member_spec', lambda *a: {})
    monkeypatch.setattr(m, '_tool', lambda key: key.lower())
    monkeypatch.setattr(m, '_stamp', lambda *a: None)
    monkeypatch.setattr(m, 'register_member', lambda *a: {'name': name})
    # The stand-in grid is bytes, not NetCDF: the static step sizes its 30" supersample from the real grid
    # (mesh_point.thirty_second_supersample, moving-subkm c7a0fd32), so this recovery test fixes it at 1.
    from woof.hex import mesh_point
    monkeypatch.setattr(mesh_point, 'thirty_second_supersample', lambda grid: 1)
    monkeypatch.setattr(init_sources, 'load_table', lambda: {})
    monkeypatch.setattr(init_sources, 'resolve_source', lambda *a: {})
    monkeypatch.setattr(init_sources, 'Cycle', SimpleNamespace(parse=lambda x: x))
    monkeypatch.setattr(init_sources, 'init_argv', lambda *a, **kw: ['init', '--out', str(kw['out'])])
    calls = []
    def mesh(spec, grid, receipt, log, env):
        calls.append('mesh'); grid.write_bytes(b'grid'); receipt.write_text('{}'); return .1
    def run(argv, log, env):
        kind = 'init' if 'init' in argv else 'static'
        calls.append(kind)
        Path(argv[argv.index('--out')+1]).write_bytes(kind.encode())
        if '--receipt' in argv: Path(argv[argv.index('--receipt')+1]).write_text('{}')
        return .1
    monkeypatch.setattr(m, 'generate_mesh', mesh)
    monkeypatch.setattr(m, '_run_logged', run)
    for kind, receipt in [('grid', 'mesh'), ('static', 'static'), ('init', 'init')]:
        (tmp_path/f'{name}-{receipt}.json').write_text('{}')
        if kind != missing: (tmp_path/f'{name}.{kind}.nc').write_bytes(kind.encode())
    assert m.prepare_member(tmp_path/'case.json', tmp_path, name, 0., 0., log=lambda _:None) == {'name': name}
    assert calls == expected
    assert all((tmp_path/f'{name}.{kind}.nc').is_file() for kind in ('grid','static','init'))
