"""Require complete expected histories and identical array bytes, shapes and types."""
import argparse,json
from pathlib import Path

def compare(left, right, expected):
    def index(rows):
        result={row['file']:row['arrays'] for row in rows}
        if len(result)!=len(rows): raise ValueError('Duplicate history filenames')
        return result
    a,b=index(left),index(right)
    defects=[]
    if len(a)!=expected or len(b)!=expected:
        defects.append(f'Expected {expected} frames, received {len(a)} and {len(b)}')
    if a.keys()!=b.keys(): defects.append('History filenames differ')
    for filename in a.keys() & b.keys():
        if a[filename].keys()!=b[filename].keys(): defects.append(f'{filename}: array inventory differs')
        for field in a[filename].keys() & b[filename].keys():
            if a[filename][field]!=b[filename][field]: defects.append(f'{filename}: {field} differs')
    return dict(bitwise_equal=not defects,expected_frames=expected,left_frames=len(a),right_frames=len(b),defects=defects)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--expected-frames',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=compare(json.loads(a.left.read_text()),json.loads(a.right.read_text()),a.expected_frames)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['bitwise_equal'] else 1)
