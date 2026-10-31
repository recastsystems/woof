"""Qualify an unapplied fork auditor proposal while preserving existing inventory."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType


def load_auditor(path, tree):
    # Both complete auditors resolve the same supplied tree, including defaults
    # captured at definition time. Only the proposed source bytes differ.
    name = 'rrfs_citation_' + hashlib.sha256(str(path).encode()).hexdigest()
    module = ModuleType(name)
    sys.modules[name] = module
    namespace = module.__dict__
    namespace['__file__'] = str(tree/'tools/check_mpas_citations.py')
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    return namespace


def inventory(module):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert module['main'](['--inventory']) == 0
    return json.loads(output.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True, type=Path)
    parser.add_argument('--patch', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    tree = checkout/'tree'
    original = tree/'tools/check_mpas_citations.py'
    assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True).strip()
    args.work.mkdir(parents=True, exist_ok=False)
    work = args.work.resolve()
    target = work/'tree/tools/check_mpas_citations.py'
    target.parent.mkdir(parents=True)
    source = original.read_bytes()
    target.write_bytes(source)
    subprocess.run(['git', 'apply', str(args.patch.resolve())], cwd=work, check=True)
    baseline = load_auditor(original, tree)
    proposal = load_auditor(target, tree)
    before, after = inventory(baseline), inventory(proposal)
    existing = {r['citation']: r for r in before if not r['citation'].startswith('MPAS-Model-rrfs/')}
    existing_after = {r['citation']: r for r in after if r['citation'] in existing}
    assert existing == existing_after, 'proposal changed existing release inventory'
    fork = [r for r in after if r['citation'].startswith('MPAS-Model-rrfs/')]
    assert fork and all('error' not in row and row['release'].startswith('RRFS ') for row in fork)
    ref = baseline['Path'](__import__('os').environ.get('MPAS_REF') or str(baseline['Path'](__import__('os').environ.get('MPAS_LANES', '.'))/'ref/MPAS-Model-v8.4.2'))
    prior_errors = baseline['check_campaign_citations'](tree/'src/hexcore', ref)
    proposed_errors = proposal['check_campaign_citations'](tree/'src/hexcore', ref)
    assert set(proposed_errors).issubset(prior_errors), 'proposal introduced an audit refusal'
    assert not any('MPAS-Model-rrfs/' in e for e in proposed_errors)
    for label, content in [('before-inventory', before), ('after-inventory', after),
                           ('before-errors', prior_errors), ('after-errors', proposed_errors)]:
        (work/(label+'.json')).write_text(json.dumps(content, indent=2)+'\n')
    assert original.read_bytes() == source
    result = dict(hex_commit=subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
                  passed=True, protected_checker_changed=False,
                  original_sha256=hashlib.sha256(source).hexdigest(),
                  patch_sha256=hashlib.sha256(args.patch.read_bytes()).hexdigest(),
                  existing_inventory_rows=len(existing), existing_inventory_identical=True,
                  authenticated_fork_rows=len(fork), before_failures=len(prior_errors),
                  after_failures=len(proposed_errors), new_failures=[])
    (work/'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
