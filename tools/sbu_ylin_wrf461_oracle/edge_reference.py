"""Rebuild the RIMI reference at the density represented by an inverse-density state."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import read_cases,read_outputs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    fixture=root/'woof/data/sbu_ylin/oracle'
    original=fixture/'sbu-ylin-cases.txt'
    cases=read_cases(original)
    density=cases['levels'][:,:,6]
    alt=np.divide(np.float32(1),density,dtype=np.float32)
    represented=np.divide(np.float32(1),alt,dtype=np.float32)
    lines=original.read_text().splitlines()
    line=1
    for col in range(cases['ncol']):
        line+=1
        for level in range(cases['nz']):
            words=lines[line].split()
            words[6]=f'{int(represented[col,level].view(np.uint32)):08x}'
            lines[line]=' '.join(words)
            line+=1
    inputs=args.build/'edge-cases.txt';inputs.write_text('\n'.join(lines)+'\n')
    outputs=args.build/'edge-declared.txt'
    command=[str(args.build/'run_sbu_ylin_declared'),str(inputs),str(outputs)]
    subprocess.run(command,check=True)
    riming=read_outputs(outputs)['levels'][:,:,6]
    words=riming.view(np.uint32)
    args.out.write_text(f"{cases['ncol']} {cases['nz']}\n"+'\n'.join(' '.join(f'{int(word):08x}' for word in row) for row in words)+'\n')
    receipt={'command':command,'density_contract':'rho = float32(1 / float32(1 / original_rho))',
             'original_cases_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
             'edge_cases_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),
             'rimi_reference_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
             'density_words_changed':int(np.count_nonzero(density.view(np.uint32)!=represented.view(np.uint32))),
             'columns':cases['ncol'],'levels':cases['nz'],'words':int(riming.size)}
    (args.build/'edge-reference-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
