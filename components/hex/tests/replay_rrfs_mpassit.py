"""Run pinned MPASSIT on native/hex history+diag pairs; Rust compares all words.

This is downstream verification orchestration, never a replacement forecast.
The same unedited workflow lists and target configuration are used on both sides.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def quoted(value):
    return "'" + str(value).replace("'", "''") + "'"


def replay(args):
    source = args.source.resolve(strict=True)
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit != '58fc18de7960e0f8845cdc56bee7fce65fbf1b06':
        raise ValueError('MPASSIT checkout differs from pinned 58fc18d')
    subprocess.run(['git','-C',str(source),'diff','--exit-code'],check=True,capture_output=True)
    workflow = args.workflow.resolve(strict=True)
    workflow_commit = subprocess.check_output(
        ['git','-C',str(workflow),'rev-parse','HEAD'],text=True).strip()
    if not workflow_commit.startswith('08eee4ce'):
        raise ValueError('workflow checkout differs from pinned 08eee4ce')
    subprocess.run(['git','-C',str(workflow),'diff','--exit-code'],check=True,capture_output=True)
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=False)
    target = args.target_namelist.read_text()
    # The target file is a config-group body; complete groups or paths must not
    # replace the verified input bindings below.
    if '&' in target or '/' in target:
        raise ValueError('target configuration must contain only config-group assignments')
    forbidden = ('grid_file_input_grid','hist_file_input_grid','diag_file_input_grid',
                 'output_file','block_decomp_file','interp_hist','interp_diag')
    if any(key in target.casefold() for key in forbidden):
        raise ValueError('target configuration cannot replace verification input/output bindings')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    outputs = []
    attempts = []
    lists = ['diaglist','histlist_2d','histlist_3d','histlist_soil']
    for side in ('native','hex'):
        run = work / side
        run.mkdir()
        for name in lists:
            shutil.copyfile(workflow/'fix/mpassit'/name,run/name)
        history = getattr(args, side+'_history').resolve(strict=True)
        diag = getattr(args, side+'_diag').resolve(strict=True)
        output = run/'mpassit.nc'
        bindings = dict(grid_file_input_grid=args.grid.resolve(strict=True),
                        hist_file_input_grid=history, diag_file_input_grid=diag,
                        output_file=output, block_decomp_file='NULL')
        namelist = run/'namelist.input'
        namelist.write_text('&config\n'+target+'\n'+''.join(
            '  '+key+'='+quoted(value)+'\n' for key,value in bindings.items())+
            '  interp_hist=.true.\n  interp_diag=.true.\n/\n')
        # Preserve the lane's inherited taskset instead of letting OpenMPI
        # choose a binding outside cores 8-23.
        argv = [str(args.mpi.resolve(strict=True)),'-np','1','--bind-to','none',
                str(args.binary.resolve(strict=True)),namelist.name]
        with (run/'stdout.log').open('w') as log:
            status = subprocess.run(argv,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        attempts.append(dict(side=side,argv=argv,exit_status=status,
            input_history_sha256=sha(history),input_diag_sha256=sha(diag),
            output_sha256=sha(output) if output.exists() else None,
            stdout=str(run/'stdout.log')))
        outputs.append(output)
        if status or not output.is_file():
            break
    compared = None
    if len(attempts)==2 and all(a['exit_status']==0 and a['output_sha256'] for a in attempts):
        comparison = subprocess.run([str(args.comparator.resolve(strict=True)),
            *map(str,outputs)],text=True,capture_output=True)
        result=json.loads(comparison.stdout) if comparison.stdout.strip() else {}
        compared = dict(exit_status=comparison.returncode,result=result,stderr=comparison.stderr)
    passed = (compared is not None and compared['exit_status']==0 and
        compared['result'].get('schema')=='mpas.native-word-compare/v1' and
        compared['result'].get('all_identical') is True and
        compared['result'].get('words',0)>0 and
        compared['result'].get('differing_words')==0 and
        not compared['result'].get('errors'))
    receipt = dict(schema='mpas.rrfs-mpassit-pair/v1',process_count=1,
        run_host=os.uname().nodename,mpassit_commit=commit,workflow_commit=workflow_commit,
        controller_cpu_affinity=sorted(os.sched_getaffinity(0)), mpi_binding='none',
        mpassit_binary_sha256=sha(args.binary),
        comparator_sha256=sha(args.comparator),target_configuration_sha256=sha(args.target_namelist),
        grid_sha256=sha(args.grid),list_sha256={name:sha(workflow/'fix/mpassit'/name) for name in lists},
        attempts=attempts,comparison=compared,all_output_words_identical=passed,
        whole_model_proof=False)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','workflow','binary','mpi','comparator','grid','native-history','native-diag',
                'hex-history','hex-diag','target-namelist','work-root','receipt'):
        parser.add_argument('--'+key,type=Path,required=True)
    raise SystemExit(replay(parser.parse_args()))
