"""The real coupled CLI preserves all checkpoint and history words on resume."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.gpu
def test_coupled_mskf_cli_resume_matches_continuous(tmp_path):
    source=os.environ.get('WOOF_MSKF_RESTART_INPUT')
    if not source:pytest.skip('provide the coupled public MSKF input pair')
    commands=[]
    def run(label,end,restart=None):
        cmd=[sys.executable,'-m','woof.cli','run','--wrfinput',source,'--run-seconds',str(end),'--outdir',str(tmp_path/label),'--allow-shared-gpu','--no-supervise']
        if restart:cmd+=['--restart',str(restart)]
        with (tmp_path/(label+'.log')).open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'command':cmd,'rc':p.returncode})
        assert p.returncode==0,(tmp_path/(label+'.log')).read_text()[-6000:]
        def elapsed(path):
            with np.load(path,allow_pickle=False) as d:return json.loads(bytes(d['__gpuwm_restart_header__']))['elapsed_seconds']
        checkpoints=list((tmp_path/label).glob('gpuwmrst*.npz'))
        assert checkpoints,'no checkpoint was written'
        result=max(checkpoints,key=elapsed);assert elapsed(result)==end
        return result
    full=run('continuous',600);mid=run('split',300);resumed=run('resumed',600,mid)
    def arrays(path):
        with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files if k!='__gpuwm_restart_header__'}
    a,b=arrays(full),arrays(resumed);assert set(a)==set(b)
    hashes={};differences=[]
    for key in sorted(a):
        if a[key].tobytes()!=b[key].tobytes():differences.append(key)
        if np.issubdtype(a[key].dtype,np.number):assert np.isfinite(a[key]).all(),key
        hashes[key]={'continuous':hashlib.sha256(a[key].tobytes()).hexdigest(),'resumed':hashlib.sha256(b[key].tobytes()).hexdigest()}
    report={'arithmetic':os.environ.get('GPUWM_WRF_EXACT','0'),'commands':commands,'arrays':len(a),'different_arrays':differences,'hashes':hashes}
    receipt=os.environ.get('WOOF_MSKF_RESTART_RECEIPT')
    if receipt:Path(receipt).write_text(json.dumps(report,indent=2))
    assert not differences,differences
    from woof.netcdf_bridge import open_dataset
    records=[]
    for frame in sorted((tmp_path/'resumed/wrfout').glob('wrfout_d01*')):
        other=tmp_path/'continuous/wrfout'/frame.name
        assert other.is_file(),frame.name
        with open_dataset(other) as x,open_dataset(frame) as y:
            assert set(x.variables)==set(y.variables)
            for field in sorted(x.variables):
                u=np.ascontiguousarray(x.variables[field][...]);v=np.ascontiguousarray(y.variables[field][...])
                assert u.tobytes()==v.tobytes(),(frame.name,field)
                if np.issubdtype(u.dtype,np.number):assert np.isfinite(u).all(),field
                records.append({'frame':frame.name,'field':field,'sha256':hashlib.sha256(u.tobytes()).hexdigest()})
    assert records,'no resumed history fields compared'
    report['history_records']=records
    if receipt:Path(receipt).write_text(json.dumps(report,indent=2))
    from test_restart import _assert_restart_equal
    _assert_restart_equal(full,resumed)
