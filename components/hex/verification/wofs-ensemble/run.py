"""The queue command runs only the ensemble forecast door, never preparation."""
from pathlib import Path
import datetime as dt
import json, os, subprocess, sys
from woof.hex import card_queue
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))

def main(phase):
    if phase not in ('pilot','showcase'): raise ValueError('unknown phase')
    if not (W/'evidence/inputs-ready.json').is_file(): raise ValueError('inputs are not prepared')
    grant=card_queue.granted()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if not grant or len(visible)!=len(grant): raise ValueError('card grant and visible devices disagree')
    members=1 if phase=='pilot' else 8
    hours=2 if phase=='pilot' else 6
    command=[sys.executable,'-m','woof.hex.cli','ensemble','--manifest',str(W/'prepared/ensemble.json'),
        '--out',str(W/'runs'/phase),'--members',str(members),'--resolution-km','3','--forecast-only',
        '--members-per-card','1','--cycle-hours',str(hours),'--history-every-minutes','60',
        '--sample-seconds','1','--gpuwm-checkout',os.environ['E']]
    (W/'runs').mkdir(exist_ok=True)
    exit_path = W/'evidence'/f'{phase}-exit.json'
    attempt = dict(status='running',returncode=None,started_utc=dt.datetime.now(dt.UTC).isoformat())
    exit_path.write_text(json.dumps(attempt)+'\n')
    (W/'evidence'/f'{phase}-grant.json').write_text(json.dumps(dict(grant=grant,visible_devices=visible,
        members=members,hours=hours,command=command,whole_card=card_queue.get(card_queue.SHARE, None)),
        indent=2)+'\n')
    result=subprocess.run(command)
    attempt.update(status='passed' if result.returncode==0 else 'failed',returncode=result.returncode,
                   finished_utc=dt.datetime.now(dt.UTC).isoformat())
    exit_path.write_text(json.dumps(attempt)+'\n')
    return result.returncode

if __name__=='__main__': sys.exit(main(sys.argv[1]))
