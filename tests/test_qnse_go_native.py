"""The QNSE pair reaches go through the stock WRF input contract."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import pytest
from woof.go_cli import _add_go_arguments,go_main,GoRefusal


def args(*words):
    parser=argparse.ArgumentParser()
    _add_go_arguments(parser)
    return parser.parse_args(list(words))


def test_native_go_dry_run_uses_the_native_importer(monkeypatch,capsys):
    import woof.wrfinput_door as door
    seen=[]
    monkeypatch.setattr(door,'resolve_wrfinput_run',lambda path: seen.append(path) or
        SimpleNamespace(experiment=SimpleNamespace(domains=(1,),run_seconds=10800.,
            start_time=datetime(2000, 1, 24, 12)),grid_ids=[1],
            namelist_input=Path('native/namelist.input')))
    assert go_main(args('--wrfinput','native','--dry-run'))==0
    assert seen==[Path('native')]
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['route']=='native-wrf'
    assert receipt['run_seconds']==10800.
    assert receipt['grid_ids']==[1]
    assert receipt['namelist_input']=='native/namelist.input'


def test_native_go_refuses_ignored_source_controls():
    with pytest.raises(GoRefusal,match='do not consume --source'):
        go_main(args('--wrfinput','native','--source','gfs','--dry-run'))
