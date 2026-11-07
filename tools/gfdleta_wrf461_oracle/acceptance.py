"""Explicit full-port acceptance gate, separate from staged primitive tests."""
import argparse
import json
from pathlib import Path
from validate_gfdleta import check_coverage, compare_words, read_words

ROOT=Path(__file__).parents[2]

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    registry=json.loads((ROOT/'woof/physics_registry_v2.json').read_text())
    options=registry['components']['radiation']['options']
    admitted=[k for k,v in options.items() if v.get('implemented') is True and v.get('selectors',{}).get('ra_lw_physics')==99 and v.get('selectors',{}).get('ra_sw_physics')==99]
    result={'pass':False,'admitted_options':admitted}
    if not admitted:
        result['error']='Radiation 99 has no implemented runtime registry row. Staged initialization and interpolation kernels do not produce forecast LW/SW tendencies or fluxes.'
    elif args.reference is None or args.candidate is None:
        result['error']='Full-port acceptance needs reference and candidate directories containing all 64 column fields and initialization tables.'
    else:
        try:
            check_coverage(args.reference);check_coverage(args.candidate)
            checks={}
            for n in (12,24,42,60):
                for arm in (0,1):
                    for kind in ('fields','tables'):
                        name=f'n{n}-a{arm}-{kind}.csv'
                        checks[name]=compare_words(read_words(args.reference/name),read_words(args.candidate/name))
            result['checks']=checks;result['pass']=all(x['pass'] for x in checks.values())
        except (OSError,ValueError) as exc:
            result['error']=str(exc)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':result['pass'],'error':result.get('error')}))
    return 0 if result['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
