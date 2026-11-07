"""Field hashes and finite ranges without carrying model arrays off a box."""
import argparse, hashlib, json
import numpy as np
from compare import read, INIT, OUT, S2

def receipt(meta, batch, path):
    init, steps = read(path, meta[batch])
    fields = {f'init.{name}': init[i] for i, name in enumerate(INIT)}
    for step, (volume, surface) in enumerate(steps, 1):
        fields.update({f'step{step}.{name}': volume[i] for i, name in enumerate(OUT)})
        fields.update({f'step{step}.{name}': surface[i] for i, name in enumerate(S2)})
    result = {'batch': batch, 'columns': len(meta[batch]['columns']), 'fields': {}}
    for name, value in fields.items():
        finite = value[np.isfinite(value)]
        result['fields'][name] = dict(sha256=hashlib.sha256(value.astype('<f4').tobytes()).hexdigest(), words=value.size, nonfinite=int(value.size-finite.size), minimum=float(finite.min()) if finite.size else None, maximum=float(finite.max()) if finite.size else None)
    return result

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('meta'); ap.add_argument('batch'); ap.add_argument('data'); ap.add_argument('output'); args=ap.parse_args()
    with open(args.output,'w') as f: json.dump(receipt(json.load(open(args.meta)),args.batch,args.data),f,indent=2)
