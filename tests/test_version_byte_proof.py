"""The version allowance must reject every unrelated container or payload edit."""
from pathlib import Path
import numpy as np
import pytest
from woof.io import nc_writer_bridge as bridge
from tools.release.default_output_proof import compare_folders
from tools.release.version_byte_proof import compare_container

pytestmark = pytest.mark.skipif(bridge.unavailable_reason() is not None,
                               reason='Rust history writer is required')


def _frame(folder, version, fmt, *, title='Identity', changed_word=False, git=None):
    folder.mkdir(exist_ok=True)
    schema = bridge.ClassicSchema(container=fmt if fmt != 'hdf' else 'cdf2')
    t = schema.def_dim('Time', 0, unlimited=True)
    y = schema.def_dim('y', 2)
    x = schema.def_dim('x', 3)
    schema.put_global_attr('TITLE', title)
    if version is not None:
        schema.put_global_attr('GPUWM_VERSION', version)
    if git is not None:
        schema.put_global_attr('git_version', git)
    schema.put_global_attr('DT', np.float32(12))
    v = schema.def_var('T', 'f4', (t, y, x))
    schema.put_var_attr(v, 'units', 'K')
    path = folder / 'wrfout_d01_2026-10-31_12:00:00'
    data = np.arange(6, dtype='f4').reshape(2, 3)
    if changed_word:
        data.view('u4')[1, 1] += 1
    create = schema.create_netcdf4 if fmt == 'hdf' else schema.create
    with create(path) as writer:
        writer.write_record(0, v, data)
        writer.write_record(1, v, data + np.float32(1))
    return path


@pytest.mark.parametrize('fmt', ['hdf', 'cdf1', 'cdf2', 'cdf5'])
@pytest.mark.parametrize('version', ['3.0.0', '3.0.0-candidate-long'])
def test_truthful_versions_are_the_only_allowed_rewrite(tmp_path, fmt, version):
    a = _frame(tmp_path/'base', '2.9.11', fmt)
    b = _frame(tmp_path/'tip', version, fmt)
    proof = compare_container(a, b, version)
    assert proof['version_only'], proof
    assert proof['differing_offsets']
    assert sorted(o for group in proof['differing_offset_groups'] for o in group['offsets']) == proof['differing_offsets']
    assert compare_folders(a.parent, b.parent)['verdict'] == 'IDENTICAL'


@pytest.mark.parametrize('fmt', ['hdf', 'cdf2'])
@pytest.mark.parametrize('edit', ['word', 'attribute', 'header'])
def test_changed_data_attributes_and_header_bytes_are_real_reds(tmp_path, fmt, edit):
    a = _frame(tmp_path/'base', '2.9.11', fmt)
    b = _frame(tmp_path/'tip', '3.0.0', fmt,
               title='Changed' if edit == 'attribute' else 'Identity',
               changed_word=edit == 'word')
    if edit == 'header':
        data = bytearray(b.read_bytes()); data[7] ^= 1; b.write_bytes(data)
    proof = compare_container(a, b, '3.0.0')
    assert not proof['version_only'], proof
    if edit != 'header':
        assert compare_folders(a.parent, b.parent)['verdict'] == 'DIFFERENT'


# PREP-LINEAGE-01: git_version, the producer commit the Rust prep and MPAS
# init/LBC writers stamp, is the second producer stamp R1 allows.
BASE_GIT = 'gpuwm-4123f1916e3c95984862cd9b21f2fc1e8dd5f850'
TIP_GIT = 'gpuwm-51bb93111ee25f79acfcb38aa57610af9d518f93'


@pytest.mark.parametrize('fmt', ['hdf', 'cdf1', 'cdf2', 'cdf5'])
@pytest.mark.parametrize('tip_git', [TIP_GIT, 'gpuwm-51bb9311', TIP_GIT + '-dirty'])
def test_a_git_version_stamp_alone_is_metadata(tmp_path, fmt, tip_git):
    a = _frame(tmp_path/'base', None, fmt, git=BASE_GIT)
    b = _frame(tmp_path/'tip', None, fmt, git=tip_git)
    proof = compare_container(a, b, stamps={'git_version': tip_git})
    assert proof['version_only'], proof
    assert proof['stamps'] == ['git_version']
    assert sorted(o for group in proof['differing_offset_groups'] for o in group['offsets']) == proof['differing_offsets']
    # The file-pair tool reads the stamps from the files themselves.
    from tools.release.version_byte_proof import main
    assert main([str(a), str(b)]) == 0


@pytest.mark.parametrize('fmt', ['hdf', 'cdf2'])
def test_both_producer_stamps_together(tmp_path, fmt):
    a = _frame(tmp_path/'base', '2.9.11', fmt, git=BASE_GIT)
    b = _frame(tmp_path/'tip', '3.0.0', fmt, git=TIP_GIT)
    proof = compare_container(a, b, '3.0.0', stamps={'git_version': TIP_GIT})
    assert proof['version_only'], proof
    assert compare_folders(a.parent, b.parent)['verdict'] == 'IDENTICAL'


@pytest.mark.parametrize('fmt', ['hdf', 'cdf2'])
@pytest.mark.parametrize('edit', ['word', 'attribute'])
def test_a_git_version_stamp_does_not_cover_anything_else(tmp_path, fmt, edit):
    a = _frame(tmp_path/'base', None, fmt, git=BASE_GIT)
    b = _frame(tmp_path/'tip', None, fmt, git=TIP_GIT,
               title='Changed' if edit == 'attribute' else 'Identity',
               changed_word=edit == 'word')
    assert not compare_container(a, b, stamps={'git_version': TIP_GIT})['version_only']
    from tools.release.version_byte_proof import main
    assert main([str(a), str(b)]) == 1


def test_only_the_named_producer_stamps_are_allowed(tmp_path):
    a = _frame(tmp_path/'base', '3.0.0', 'cdf2', title='One')
    b = _frame(tmp_path/'tip', '3.0.0', 'cdf2', title='Two')
    proof = compare_container(a, b, stamps={'TITLE': 'Two'})
    assert not proof['version_only'] and 'not producer stamps' in proof['reason']
