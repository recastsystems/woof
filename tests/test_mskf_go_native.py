"""The stock-WRF go route binds checkpoint selection and retention."""
import argparse
from woof import go_cli


def test_native_go_forwards_checkpoint_controls(tmp_path,monkeypatch):
    from woof import wrfinput_go
    parser=argparse.ArgumentParser();go_cli._add_go_arguments(parser)
    args=parser.parse_args(['--wrfinput',str(tmp_path),'--native-shared-gpu',
        '--checkpoint-at','end','--keep-checkpoints','2','--products','none'])
    seen={}
    def launch(received, **kwargs):
        seen.update(vars(received))
        assert received.wrfinput == tmp_path
        return 0
    monkeypatch.setattr(wrfinput_go, 'launch', launch)
    assert go_cli._go_wrfinput(args) == 0
    assert seen['checkpoint_at'] == 'end'
    assert seen['allow_shared_gpu'] is True
    assert seen['keep_checkpoints'] == 2
    assert seen['render_products'] == 'none'
