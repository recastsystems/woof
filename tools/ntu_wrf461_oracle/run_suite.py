"""Run column batches in one process so each arithmetic arm compiles once."""
import argparse,json
from pathlib import Path
from run_gpu import main as run
from receipt import receipt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('inputs',type=Path); ap.add_argument('outputs',type=Path); ap.add_argument('--batch',action='append'); args=ap.parse_args()
    args.outputs.mkdir(parents=True,exist_ok=True)
    meta=json.loads((args.inputs/'ntu_batches.json').read_text())
    for name in args.batch or list(meta):
        output=args.outputs/f'gpu-{name}.bin'
        run(args.inputs/f'ntu_batch_{name}.bin',output)
        (args.outputs/f'gpu-{name}.json').write_text(json.dumps(receipt(meta,name,output),indent=2))
        print(f'completed {name}: {len(meta[name]["columns"])} columns',flush=True)

if __name__=='__main__': main()
