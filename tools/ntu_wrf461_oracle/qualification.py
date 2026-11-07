"""Check exact causal closure and report each fixture field's error envelope."""
import argparse,csv,json
from pathlib import Path
from tools.ntu_wrf461_oracle.compare import INIT,OUT,S2
def grade(root):
    rows=[];batches={};errors=[]
    for b in 'ABCDE':
        c='independent-control-'+b+('-clean' if b=='E' else '')
        control=json.loads((root/(c+'.json')).read_text())
        steps=3 if b in 'AB' else 2
        expected={f'init.{n}' for n in INIT}|{
            f'step{k}.{n}' for k in range(1,steps+1) for n in OUT+S2}
        if set(control['fields'])!=expected:errors.append(b+': incomplete control inventory')
        lookup=json.loads((root/(c+'-lookup.json')).read_text())
        if lookup['lookup_misses']!=[0,0]:errors.append(b+': missed intrinsic inputs')
        if any(x['bit_mismatches'] or x['nonfinite'] for x in control['fields'].values()):
            errors.append(b+': unexplained output divergence')
        arms=json.loads((root/f'own-erf-arms-{b}.json').read_text())
        if any(x['bit_mismatches'] or x['nonfinite'] for x in arms['fields'].values()):
            errors.append(b+': default and strict differ')
        counts={}
        for arm in ('strict','default'):
            result=json.loads((root/f'independent-{arm}-{b}.json').read_text())
            if set(result['fields'])!=set(control['fields']):errors.append(b+': field inventory mismatch')
            counts[arm]=sum(x['bit_mismatches'] for x in result['fields'].values())
            for field,x in result['fields'].items():
                if x['nonfinite']:errors.append(b+'/'+field+': nonfinite')
                rows.append(dict(batch=b,arm=arm,field=field,bit_mismatches=x['bit_mismatches'],
                    maximum_ulp=x['max_ulp'],absolute_envelope=x['max_abs_error'],
                    cause='ERF rounding with exact counterfactual closure' if x['bit_mismatches'] else 'exact',
                    counterfactual_bit_mismatches=control['fields'].get(field,{}).get('bit_mismatches')))
        batches[b]=dict(columns=control['columns'],changed_words=counts,lookup_misses=lookup['lookup_misses'])
    for arm in ('strict','default'):
        r=json.loads((root/f'runtime-erf-production-{arm}.json').read_text())
        for name in ('erff','erf'):
            if r[name]['max_ulp']>1 or not r[name]['special_words_match']:errors.append(arm+'/'+name+': primitive contract failed')
    r=json.loads((root/'own-erf-mutation-compare.json').read_text())
    if not any(x['bit_mismatches'] for x in r['fields'].values()):errors.append('mutation was not caught')
    if sum(x['columns'] for x in batches.values())<64:errors.append('fewer than 64 columns')
    return dict(passed=not errors,grade='declared-ERF-departures' if not errors else 'fail',batches=batches,errors=errors,
                envelope_scope='observed oracle errors on these fixtures, not forecast acceptance thresholds'),rows
def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipts',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    result,rows=grade(a.receipts);a.output.write_text(json.dumps(result,indent=2))
    with a.output.with_suffix('.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
