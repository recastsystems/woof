"""Native stream layout is Fortran cells(level,column,field), not C axes."""
import struct
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_native_capture import read_capture


def _capture():
    header=struct.pack('<4if6i10f',1,2,3,1,15.,5,1,2,18,6,6,
                       1.e7,0.,1.,4.e5,4.e4,8.e5,3.e6,500.,900.,100.)
    coords=np.array([[17,31],[18,32]],'<i4')
    cells=np.arange(29*2*3,dtype='<f4').reshape(29,2,3)
    surface=np.arange(9*2,dtype='<f4').reshape(9,2)
    return (header+coords.tobytes()+cells.tobytes()+surface.tobytes()+
            (cells+1000).tobytes()+(surface+2000).tobytes())


def test_fortran_capture_axes_and_both_snapshots(tmp_path):
    path=tmp_path/'capture.bin'; path.write_bytes(_capture())
    meta,(before,after)=read_capture(path)
    assert (meta['nx'],meta['nz'],meta['itimestep'],meta['dt'])==(2,3,1,15.)
    assert meta['coords']==[[17,31],[18,32]]
    np.testing.assert_array_equal(before[0]['qv'],[[6,7,8],[9,10,11]])
    np.testing.assert_array_equal(after[0]['pressure'],[[1120,1121,1122],[1123,1124,1125]])
    np.testing.assert_array_equal(after[1]['sr'],[2016,2017])


@pytest.mark.parametrize('change',[lambda b:b[:-4],lambda b:b+b'\0'*4,
                                    lambda b:struct.pack('<i',99)+b[4:]])
def test_incomplete_or_unknown_native_stream_is_refused(tmp_path,change):
    path=tmp_path/'capture.bin'; path.write_bytes(change(_capture()))
    with pytest.raises(ValueError): read_capture(path)


def test_complete_native_row_keeps_full_group_context(tmp_path):
    nx,nz=75,3
    header=struct.pack('<4if6i10f',2,nx,nz,114,12.,5,1,2,18,7,2,
                       1.e7,0.,1.,4.e5,4.e4,8.e5,3.e6,500.,900.,100.)
    coords=np.column_stack((np.arange(1,nx+1),np.full(nx,2))).astype('<i4')
    cells=np.arange(29*nx*nz,dtype='<f4').reshape(29,nx,nz)
    surface=np.arange(9*nx,dtype='<f4').reshape(9,nx)
    raw=header+coords.tobytes()+2*(cells.tobytes()+surface.tobytes())
    path=tmp_path/'row.bin';path.write_bytes(raw)
    meta,(before,after)=read_capture(path)
    assert meta['complete_native_row'] and meta['capture_version']==2
    assert meta['nx']==75 and before[0]['theta'].shape==(75,3)
    # Missing/reordered X or crossing Y would change native wet-64 grouping.
    for changed in (coords[::-1],coords+np.column_stack((np.zeros(nx),np.arange(nx))).astype('<i4')):
        path.write_bytes(header+changed.tobytes()+2*(cells.tobytes()+surface.tobytes()))
        with pytest.raises(ValueError,match='contiguous X'):read_capture(path)
