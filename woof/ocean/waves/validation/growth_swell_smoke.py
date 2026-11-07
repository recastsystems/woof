"""One pool invocation and 600 s outer timeout for growth and source-free swell."""
import json
from pathlib import Path
import subprocess
import sys
import time
import cupy as cp
from . import published_growth, swell_packet, spherical_packet


def main():
    output=Path(sys.argv[1]);resume='--resume-preflight' in sys.argv
    prior=None
    if resume:
        prior=json.loads((output/'receipt.json').read_text())
        if prior['status']!='ERROR' or (output/'growth').exists() or (output/'spherical/receipt.json').exists():
            raise RuntimeError('Resume is allowed only before spherical/growth integrations')
        (output/'preflight-error.json').write_text(json.dumps(prior,indent=2)+'\n')
    else:output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    receipt={'status':'RUNNING','wall_budget_seconds':600,
             'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    path=output/'receipt.json'
    def save():path.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        packet=json.loads((output/'packet/receipt.json').read_text()) if resume else swell_packet.run(output/'packet')
        if resume:receipt['preflight_resume']=prior
        receipt['packet_status']=packet['status'];save()
        cp.cuda.get_current_stream().synchronize();cp.get_default_memory_pool().free_all_blocks()
        spherical=spherical_packet.run(output/'spherical')
        receipt['spherical_packet_status']=spherical['status'];save()
        spherical_packet.charts(output/'spherical',spherical)
        cp.cuda.get_current_stream().synchronize();cp.get_default_memory_pool().free_all_blocks()
        remaining=min(570.,580.-(time.perf_counter()-began))
        published_growth.run(output/'growth', budget_seconds=remaining)
        growth=json.loads((output/'growth/receipt.json').read_text())
        receipt['duration_measurements']=[]
        for item in growth['results']:
            if item['mode']=='duration':
                normalized=item['hourly'][-1]['hs_m']*9.80665/item['wind_ms']**2
                receipt['duration_measurements'].append({'pack':'st4','wind_ms':item['wind_ms'],
                    'normalized_hs':normalized})
        swell_packet.charts(output/'packet',packet)
        receipt['status']='DONE' if packet['status']=='PASS' and spherical['status']=='PASS' and growth['status']=='DONE' else 'FAIL'
        receipt['scope']='Source-free packet passes and completed finite growth experiment. Growth science requires review against the published spectra and fetch curves.'
        receipt['retired_gate']='A 240-hour U10-scaled PM band is not an ST4 stationary limit: 10.1175/2010JPO4324.1 p1925 explicitly describes continued growth. See validation/GROWTH-REFEREES.md.'
    except BaseException as exc:
        receipt['status']='ERROR';receipt['error']=str(exc);raise
    finally:
        receipt['wall_seconds']=time.perf_counter()-began;save()


if __name__=='__main__':main()
