"""Stage only small receipts/configuration for prompt retrieval after a run.

Run on the compute host. Does not package native arrays, history, restarts,
credentials, environments or shared input files.
"""
from pathlib import Path
import argparse,json,shutil,hashlib

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=a.out.resolve()
    if root==out or out.is_relative_to(root):raise ValueError('staging directory must be outside source run directory')
    out.mkdir(parents=True,exist_ok=True);receipt=[];total=0
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file():continue
        if path.suffix not in {'.json','.toml','.csv','.log','.txt','.wps'}:continue
        size=path.stat().st_size
        if size>2_000_000:continue
        # Configuration, metadata and small logs only. Avoid incidental array
        # text dumps; the final stage is bounded independently of file sizes.
        if total+size>20_000_000:raise ValueError('small artifact package exceeds 20 MB')
        target=out/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        data=path.read_bytes();target.write_bytes(data);total+=len(data)
        receipt.append(dict(path=str(path.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (out/'artifact-index.json').write_text(json.dumps(dict(source=str(root),files=receipt,total_bytes=total),indent=2)+'\n')
if __name__=='__main__':main()
