#!/usr/bin/env python3
"""Replay authenticated r0 stream publications; never claim a Hex forecast."""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from replay_rrfs_pool_records import replay_append

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--comparator', type=Path, required=True)
parser.add_argument('--additional-publications', type=Path)
args = parser.parse_args()
checkout = Path(__file__).resolve().parents[2]
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
# The native reference store sits beside the pinned fork checkout (RRFS_MPAS_REF). On a host
# without it, RRFS_REFERENCE_HOST (user@host) names where to copy the publications from.
authority = Path(os.environ.get('RRFS_MPAS_REF') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs/MPAS-Model-rrfs').parent/'forecasts/native-da-cycling-original-ci-v1'
remote = None if authority.is_dir() else os.environ['RRFS_REFERENCE_HOST']
if remote:
    local = output/'authority'
    subprocess.run(['rsync', '-a', remote+':'+str(authority)+'/', str(local)+'/'],check=True)
    authority = local
pub = json.loads((authority/'PUBLICATION.json').read_text())
assert sha(authority/'PUBLICATION.json') == 'f4e042a4254f2a59cec9f0484180323e78e5b1292941c8a1ec328a1c4273f61b'
for name,digest in pub['files'].items():
    assert sha(authority/name)==digest,(name,'authority differs')
index=json.loads((authority/'INDEX.json').read_text())
assert index['completed'] and index['ranks']==1
rows=[]
references = [(case['name'], phase, reference) for case in index['cases']
              for phase, reference in case['publications'].items()]
if args.additional_publications is not None:
    additional = json.loads(args.additional_publications.read_text())
    assert additional['schema'] == 'r11-additional-native-stream-publications/v1'
    references.extend((row['case'], row['phase'], row) for row in additional['publications'])
assert len({case+'-'+phase for case,phase,_ in references}) == len(references)
for case,phase,reference in references:
        label=case+'-'+phase
        assert Path(label).name == label and label not in ('.', '..')
        published=Path(reference['path'])
        native=published
        if remote:
            native=output/label/'native'
            native.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-a','--chmod=Du+w',remote+':'+str(published)+'/',str(native)+'/'],check=True)
        assert sha(native/'PUBLICATION.json')==reference['publication_sha256']
        phase_pub=json.loads((native/'PUBLICATION.json').read_text())
        if 'files' in reference:
            assert len(phase_pub['files'])==reference['files']
        for name,digest in phase_pub['files'].items():
            assert sha(native/name)==digest,(label,name,'published bytes differ')
        manifest=json.loads((native/'MANIFEST.json').read_text())
        assert manifest['completed'] and manifest['ranks']==1 and manifest['recipe']=='gnu-strict'
        assert manifest['fork']=='1d6e370e7b43a332074809f170532a216c2212fb'
        labels=manifest.get('reference_labels', [])
        derivatives=manifest.get('init_derivative_authorities', [])
        if 'reference_labels' in reference:
            assert labels == reference['reference_labels']
            assert derivatives == reference['init_derivative_authorities']
        target=output/label/'session'
        receipt=output/(label+'.json')
        command=[sys.executable,str(checkout/'tree/tests/replay_rrfs_session.py'),
                 '--native-run',str(native),'--output',str(target),
                 '--comparator',str(args.comparator.resolve()),
                 '--receipt',str(receipt)]
        with (output/(label+'.log')).open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        row=dict(case=case,phase=phase,publication_path=str(published),
                 publication_sha256=reference['publication_sha256'],exit_status=status,
                 receipt=receipt.name,whole_model_proof=False,
                 reference_labels=labels,init_derivative_authorities=derivatives)
        if receipt.is_file():
            session=json.loads(receipt.read_text())
            words=json.loads(receipt.with_suffix('.words.json').read_text())
            row.update(passed=session['passed'],attributes_identical=session['all_global_attributes_identical'],
                       files=len(words['files']),
                       words=sum(f['comparison']['words'] for f in words['files']),
                       differing_words=sum(f['comparison']['differing_words'] for f in words['files']),
                       session_receipt_sha256=sha(receipt),words_receipt_sha256=sha(receipt.with_suffix('.words.json')))
        if status == 0:
            append_output = output / label / 'append'
            append_output.mkdir()
            append_receipt = output / (label + '.append.json')
            import os
            appended = replay_append(native, append_output, append_receipt,
                                     Path(os.environ['RRFS_NATIVE_STREAM_LIBRARY']))
            row.update(append_passed=appended['passed'], append_words=appended['words'],
                       append_receipt_sha256=sha(append_receipt))
        rows.append(row)
        (output/'attempts.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(label,status,flush=True)
        # Retain all hashes/comparisons and metadata; retire only our produced
        # session files and private host source copies, never reference files.
        retired=[]
        if status==0:
            dirs=[target, append_output]+([native] if native!=published else [])
            for directory in dirs:
                for p in directory.rglob('*'):
                    if p.is_file() and not p.is_symlink() and p.name.endswith(('.nc','.nc.done')):
                        retired.append(dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p)))
                        p.unlink()
            (output/(label+'.retirement.json')).write_text(json.dumps(retired,indent=2)+'\n')
result=dict(schema='r11-current-r0-original-da-session-matrix/v1',
            scope='Complete native-state stream session replay on authenticated r0 publications; no Hex atmosphere executed',
            process_count=1,host=socket.gethostname(),whole_model_proof=False,
            publication_sha256=sha(authority/'PUBLICATION.json'),
            hex_commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),
            
            comparator_sha256=sha(args.comparator),
            session_source_sha256=sha(checkout/'tree/tests/replay_rrfs_session.py'),
            wrapper_sha256=sha(Path(__file__)),
            additional_publications_sha256=sha(args.additional_publications) if args.additional_publications else None,
            reference_labels=sorted({label for row in rows for label in row['reference_labels']}),
            init_derivative_authorities=[authority for row in rows for authority in row['init_derivative_authorities']],
            cases=rows,completed_phases=len(rows),expected_phases=len(references),
            files=sum(r.get('files',0) for r in rows),words=sum(r.get('words',0) for r in rows),
            differing_words=sum(r.get('differing_words',0) for r in rows),
            append_words=sum(r.get('append_words', 0) for r in rows),
            append_passed=all(r.get('append_passed') for r in rows),
            passed=len(rows)==len(references) and all(r['exit_status']==0 and r.get('passed') and r.get('attributes_identical') and r.get('differing_words')==0 and r.get('files',0)>0 for r in rows))
(output/'matrix.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('completed_phases','files','words','differing_words','passed')}))
raise SystemExit(0 if result['passed'] else 1)
