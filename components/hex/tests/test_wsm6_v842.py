"""Whole-module oracle tests. Build with tests/wsm6_oracle/prepare.py first."""
from pathlib import Path
import ctypes as ct
import importlib.util
import os
import sys
import numpy as np
import pytest
from woof.hex.mpasphys.wsm6_v842 import cuda_source

ORACLE=Path(__file__).with_name('wsm6_oracle')


def _helper(name):
    spec=importlib.util.spec_from_file_location(name,ORACLE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


@pytest.fixture(autouse=True)
def oracle_imports(monkeypatch):
    """Keep this oracle's sibling imports scoped to each WSM6 test.

    The full G0 battery can already have another module named ``replay``
    loaded. Prepending a path does not replace that cached module, so bind
    our own helper temporarily and restore the previous entry afterwards.
    """
    monkeypatch.syspath_prepend(str(ORACLE))
    monkeypatch.setitem(sys.modules, 'replay', _helper('replay'))


@pytest.fixture
def built():
    root=os.environ.get('WSM6_ORACLE_PATH')
    if root is None:pytest.skip('set WSM6_ORACLE_PATH to the standalone oracle')
    p=Path(root).resolve()
    for name in ('liboracle.so','libwrapper.so','libdriver.so','libhost.so'):
        assert (p/name).is_file(),f'build standalone oracle before replay: {name}'
    return p


def test_shared_math_is_required():
    with pytest.raises(RuntimeError,match='shared RKIND/glibc CUDA math header'):
        cuda_source('')


def test_production_math_factory_is_published():
    # THE BREAKAGE: cuda_entry asks woof.hex.rkind_libm for wsm6_device_math_header and raised
    # "shared rkind_libm.wsm6_device_math_header is not published" at the first WSM6 call (0.4.1-0.4.9).
    from woof.hex import rkind_libm
    header=rkind_libm.wsm6_device_math_header()
    for name in ('mp_exp','mp_log','mp_log10','mp_pow','mp_sqrt','mp_complex_div'):
        assert name in header
    assert 'Wsm6Workspace' in cuda_source(header)



def test_workspace_is_lane_interleaved_with_statements_unchanged():
    # THE BREAKAGE: one Wsm6Workspace struct per thread made every warp access touch 32 cache
    # lines and the launcher ran 1,024 columns per launch; mesoscale_reference ran 0.50 s/step
    # against 0.064 for convection_permitting (x1.2562 24 h fast, RTX PRO 6000, ffa87990).
    from collections import Counter
    from woof.hex import rkind_libm
    from woof.hex.mpasphys import wsm6_v842 as w
    source=cuda_source(rkind_libm.wsm6_device_math_header(),max_levels=55)
    assert 'S.workspace->' not in source and source.count('(S.workspace,WSM6_OFFSET(')==132 and source.count('__builtin_offsetof')==1
    raw=(w.CUDA_MODULES+w.CUDA_COLUMNS).splitlines()
    new=w._interleaved(w.CUDA_MODULES+w.CUDA_COLUMNS).replace(w._WS_SUPPORT,'',1).splitlines()
    moved=lambda l:('S.workspace' in l or 'WsP<' in l or 'wsm6_view<' in l or l.startswith('__device__ ')
                    or 'Wsm6Workspace' in l or 'Wsm6State state;' in l or 'column_' in l or 'production_on?' in l
                    or l==' const int wl=i-first;')
    assert Counter(l for l in raw if not moved(l))==Counter(l for l in new if not moved(l))


def test_one_launch_takes_every_column_within_the_budget():
    from woof.hex.mpasphys.wsm6_v842 import _tile_columns
    column=39448+3184  # 55 levels: interleaved arrays + native state, measured on sm_120
    assert _tile_columns(2562,column)==2592
    tile=_tile_columns(163842,column)
    assert tile%32==0 and tile*column<=1536<<20 and tile>30000

def test_complete_mmm_and_wrapper(built,tmp_path):
    assert _helper('replay').replay(built,tmp_path/'columns.json')==0


def test_native_continuing_column_boundaries(built,tmp_path,monkeypatch):
    """Compare all native words at logical one/two levels and signed timesteps.

    mp_wsm6.F90:1418-1468 includes zero-dt MIN and NaN operand semantics.
    Native output words are oracle results only, never transcription inputs.
    """
    import contextlib
    import io
    import json

    monkeypatch.syspath_prepend(str(ORACLE))
    native=_helper('native_defect_measurement')
    rows=[]
    for index in range(48):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            native.child(built,index)
        result=json.loads(stream.getvalue())
        rows.append({'case':index,'both_complete':True,
                     'repeats':[{'result':result}]})
    columns=tmp_path/'native-column-boundaries.json'
    columns.write_text(json.dumps({'complete':True,'rows':rows}))
    empty_driver=tmp_path/'no-driver-cases.json'
    empty_driver.write_text(json.dumps({'complete':True,'rows':[]}))
    receipt=tmp_path/'host-column-boundaries.json'
    _helper('replay_host_boundary_words').run(built,columns,empty_driver,receipt)
    replay=json.loads(receipt.read_text())
    assert replay['compared_words']==1488
    assert replay['differing_words']==0


def test_native_zero_timestep_driver_clamps(built,tmp_path,monkeypatch):
    """Replay complete native driver outputs through GNU's exceptional clamps.

    MPAS-A v8.4.2 mpas_atmphys_driver_microphysics.F:700-727. These two-level
    high-top fixtures keep kp+1 inside assigned native storage; every field
    word is compared, including NaN signs and each reflectivity diagnostic.
    """
    import contextlib
    import io
    import json

    monkeypatch.syspath_prepend(str(ORACLE))
    native=_helper('native_driver_defect_measurement')
    rows=[]
    for index in (58,67,76):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            native.child(built,index,(1,2,8))
        result=json.loads(stream.getvalue())
        assert result['nz']==2 and result['dt']==0 and not result['low_top']
        rows.append({'case':index,'both_complete':True,
                     'repeats':[{'result':result}]})
    driver=tmp_path/'native-driver-clamps.json'
    driver.write_text(json.dumps({'complete':True,'rows':rows}))
    empty_columns=tmp_path/'no-column-cases.json'
    empty_columns.write_text(json.dumps({'complete':True,'rows':[]}))
    receipt=tmp_path/'host-driver-clamps.json'
    _helper('replay_host_boundary_words').run(built,empty_columns,driver,receipt)
    replay=json.loads(receipt.read_text())
    assert len(replay['rows'])==3 and replay['compared_words']>200
    assert replay['differing_words']==0


def test_driver_boundary_including_halos(built,tmp_path):
    assert _helper('compare_driver').compare(built/'libdriver.so',built/'libhost.so',tmp_path/'driver.json')==0


def test_gnu_radar_initialization_words(built):
    outputs=[]
    for name,func in [('liboracle.so','radar_constants_oracle'),('libhost.so','radar_constants_host')]:
        f=getattr(ct.CDLL(str(built/name)),func);f.argtypes=[ct.POINTER(ct.c_double)]
        a=np.zeros(32,np.float64);f(a.ctypes.data_as(ct.POINTER(ct.c_double)));outputs.append(a)
    np.testing.assert_array_equal(outputs[0].view(np.uint64),outputs[1].view(np.uint64))


def test_native_radar_defect_classifier(built):
    """Locate the native unassigned kp+1 read; this never refuses a forecast."""
    f=ct.CDLL(str(built/'libhost.so')).height_guard_host
    f.argtypes=[ct.c_int]*3+[ct.POINTER(ct.c_float),ct.POINTER(ct.c_int)]
    z=np.array([[0,0,0],[100,800,100],[200,1200,200],[300,1600,300]],np.float32)
    status=np.full(3,42,np.int32)
    f(3,3,2,z.ctypes.data_as(ct.POINTER(ct.c_float)),status.ctypes.data_as(ct.POINTER(ct.c_int)))
    np.testing.assert_array_equal(status,[1,0,42])


def test_radar_alternate_configurations(built,tmp_path):
    assert _helper('radar_branches').compare(built,tmp_path/'radar.json')==0


def test_effective_radius_density_boundaries_and_finalization(built):
    constants=_helper('replay').CONSTANTS
    fp=ct.POINTER(ct.c_float);ip=ct.POINTER(ct.c_int)
    outputs=[]
    for enabled in (0,1):
        for name,func in [('liboracle.so','radii_boundary_oracle'),('libhost.so','radii_boundary_host')]:
            f=getattr(ct.CDLL(str(built/name)),func);f.argtypes=[ct.c_int,ct.c_int,fp,fp,fp,ip]
            v=np.zeros((10,8),np.float32);v[0]=[250,280]*4
            v[7]=[1e-10,1,1e-10,1,1,1,1,1]
            v[2:7]=.2;v[2:7,4:]=0
            r=np.full((3,8),.00007,np.float32);status=np.full(3,42,np.int32)
            f(8,enabled,constants.ctypes.data_as(fp),v.ctypes.data_as(fp),r.ctypes.data_as(fp),status.ctypes.data_as(ip))
            outputs.append((r.view(np.uint32).copy(),status.copy()))
        np.testing.assert_array_equal(outputs[-2][0],outputs[-1][0])
        np.testing.assert_array_equal(outputs[-2][1],outputs[-1][1])


def test_all_defined_optional_diagnostic_combinations(built,tmp_path):
    assert _helper('option_branches').compare(built,tmp_path/'optional-diagnostics.json',_helper('replay').CONSTANTS)==0


def test_private_paths_not_called_by_public_driver(built,tmp_path):
    if not (built/'libprivate.so').is_file():
        pytest.skip('prepare --expose-private for authenticated test-only private access')
    assert _helper('private_branches').compare(built,tmp_path/'private-paths.json',_helper('replay').CONSTANTS)==0


def test_no_precipitation_without_optional_surface_outputs(built):
    """mp_wsm6.F90:769: zero falls with absent optional surface outputs."""
    constants=_helper('replay').CONSTANTS;fp=ct.POINTER(ct.c_float)
    outputs=[]
    for lib,func in [('liboracle.so','optional_surface_oracle'),('libhost.so','optional_surface_host')]:
        f=getattr(ct.CDLL(str(built/lib)),func)
        f.argtypes=[ct.c_int,ct.c_float,ct.c_int,ct.c_int]+[fp]*4
        v=np.zeros((10,8),np.float32);v[0]=300
        v[7]=1.;v[8]=70000.;v[9]=500.
        s=np.full(7,1.25,np.float32);prod=np.full((2,8),42,np.float32)
        f(8,30,0,0,*[a.ctypes.data_as(fp) for a in (constants,v,s,prod)])
        assert s[6]==np.float32(0)
        outputs.append(np.concatenate([v.ravel(),s,prod.ravel()]).view(np.uint32))
    np.testing.assert_array_equal(*outputs)


def test_native_vertical_buffer_species_extents(built,tmp_path):
    assert _helper('buffer_layout_audit').audit(built,tmp_path/'buffer-layout.json') > 0
