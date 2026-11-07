"""An incomplete or structurally different history cannot pass the OFF proof."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('hash_compare',Path(__file__).parents[1]/'tools/compare_history_hashes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def frame(name='frame0.nc',digest='a'):
    return dict(file=name,arrays={'sst':dict(shape=[2],dtype='float32',sha256=digest)})

def test_missing_history_cannot_pass_even_when_existing_arrays_match():
    assert not m.compare([frame()],[frame()],2)['bitwise_equal']

def test_complete_matching_history_passes_and_changed_late_frame_fails():
    a=[frame(),frame('frame1.nc')]
    assert m.compare(a,a,2)['bitwise_equal']
    assert not m.compare(a,[frame(),frame('frame1.nc','b')],2)['bitwise_equal']

def test_duplicate_frames_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'):m.compare([frame(),frame()],[frame()],2)
