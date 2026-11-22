"""Authenticate complete generated modules against the production artifact."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from generate_port import generate


def literals(path):
    wanted={'_SUPPORT','_SCHEME_SOURCE','_ENTRY','_DRIVER','ARGUMENTS','SOURCE_HASHES','UPSTREAM_SOURCE_HASHES',
            'AUTHORITY_LABEL','AUTHORITY_PATCH_SHA256'}
    return {target.id:ast.literal_eval(node.value)
            for node in ast.parse(path.read_text()).body if isinstance(node,ast.Assign)
            for target in node.targets if isinstance(target,ast.Name) and target.id in wanted}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    production=Path('hex/tree/src/hexcore/mpasphys/rrfs/cu_gfl_bf6a4b0.py')
    generated=args.output/'generated.py'
    generate(Path('native'),generated)
    original,rebuilt=literals(production),literals(generated)
    different=[key for key in original if
               (original[key].strip() if isinstance(original[key],str) else original[key]) !=
               (rebuilt[key].strip() if isinstance(rebuilt[key],str) else rebuilt[key])]
    receipt={'scope':'complete GNU-tree regeneration, not numerical acceptance',
             'authority_label':original['AUTHORITY_LABEL'],
             'authority_patch_sha256':original['AUTHORITY_PATCH_SHA256'],
             'production_sha256':hashlib.sha256(production.read_bytes()).hexdigest(),
             'generated_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),
             'compared':sorted(original),'different':different}
    (args.output/'generation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    return bool(different)


if __name__=='__main__':raise SystemExit(main())
