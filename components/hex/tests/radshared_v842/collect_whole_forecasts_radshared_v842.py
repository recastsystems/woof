"""Bind original required requests to authenticated whole forecast attempts.

This inventory never enrolls missing words or promotes a comparison-only run.
It preserves absent features, absent decks and failed receipts explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    checkout = Path(config['checkout'])
    required_path = Path(config['required'])
    required = json.loads(required_path.read_text())
    assert required['process_count'] == 1
    def identity(row):
        return (row['mesh'], row['date'], row['process_count'],
                row['namelist_sha256'], row['streams_sha256'])
    requests = {identity(row): i for i, row in enumerate(required['requests'])}
    assert len(requests) == required['required_requests']
    decks = {}
    missing_plans = []
    for packet in config['plans']:
        plan_path = Path(packet['plan'])
        if not plan_path.exists():
            missing_plans.append(str(plan_path))
            continue
        plan = json.loads(plan_path.read_text())
        assert plan['process_count'] == 1 and plan['source_private_patch'] is False
        assert plan['clean_checkout_at_start'] is True
        for filename, expected in plan['source_sha256'].items():
            assert sha(checkout / filename) == expected
        matrix_path = Path(packet['matrix'])
        results = {}
        if matrix_path.exists():
            matrix = json.loads(matrix_path.read_text())
            assert matrix['process_count'] == 1 and matrix['plan_sha256'] == sha(plan_path)
            results = {row['deck']: row for row in matrix['rows']}
        for row in plan['decks']:
            path = Path(row['path'])
            assert sha(path) == row['sha256']
            deck = json.loads(path.read_text())
            if 'only_ids' in packet and deck['id'] not in packet['only_ids']:
                continue
            key = identity(dict(mesh=deck['mesh'], date=deck['date'],
                process_count=deck['process_count'], namelist_sha256=deck['inputs']['namelist']['sha256'],
                streams_sha256=deck['inputs']['streams']['sha256']))
            number = requests[key]
            assert number not in decks, 'One actual request cannot count twice'
            command = deck['hex_command']
            overlays = [command[i + 1] for i, word in enumerate(command[:-1]) if word == '--physics']
            assert overlays == [required['requests'][number]['hex_backend_override']]
            assert command[command.index('--physics-backend') + 1] == 'mpasphys'
            record = dict(path=str(path.resolve()), request_index=number, deck_sha256=sha(path),
                source_commit=plan['source_commit'], source_plan_sha256=sha(plan_path),
                status='whole forecast pending', accepted=False)
            if deck['id'] in results:
                actual = results[deck['id']]
                receipt_path = Path(actual['receipt'])
                assert sha(receipt_path) == actual['receipt_sha256']
                receipt = json.loads(receipt_path.read_text())
                assert receipt['deck'] == deck['id']
                assert all(receipt[name] == deck[name] for name in
                           ('pin_set', 'mesh', 'date', 'process_count'))
                refused = receipt['evidence_kind'] == 'REFUSAL'
                if refused:
                    # Preflight refusals have no model/source provenance. The
                    # pinned plan and receipt authenticate the failed request,
                    # without claiming that its model actually executed.
                    assert receipt['passed'] is False
                    assert not receipt.get('model_execution', False)
                    assert receipt.get('acceptance_tier', receipt.get('tier')) is None
                else:
                    assert receipt['provenance']['deck_sha256'] == sha(path)
                    provenance = receipt['provenance']['hex_source']
                    assert provenance['commit'] == plan['source_commit'] and not provenance['dirty']
                record.update(status='whole attempt recorded', receipt=str(receipt_path.resolve()),
                    receipt_sha256=sha(receipt_path), evidence_kind=receipt['evidence_kind'],
                    model_execution=receipt.get('model_execution', False), verifier_passed=receipt['passed'],
                    acceptance_tier=receipt.get('acceptance_tier'), counters=actual.get('counters'))
                if 'work_directory' in actual:
                    record['work_directory'] = actual['work_directory']
                if refused:
                    record.update(status='request refused; no model execution',
                                  refusal_failures=receipt['failures'])
                if receipt['evidence_kind'] == 'FORECAST' and record['model_execution']:
                    record['status'] = 'complete forecast; original acceptance retained'
                # No owner context approval is inferred from written-word zeros.
                record['accepted'] = (receipt['passed'] and actual['exit'] == 0
                    and receipt['evidence_kind'] == 'FORECAST' and record['model_execution']
                    and record['acceptance_tier'] == 'BITWISE')
            decks[number] = record
    out = dict(schema='p21-whole-model-matrix/v1', process_count=1,
        ready=False, matrix_complete=False, inventory_only=True,
        required_requests=required['required_requests'], required_matrix_sha256=sha(required_path),
        config_sha256=sha(args.config), decks=[decks[i] for i in sorted(decks)],
        unbound_requests=sorted(set(range(required['required_requests'])) - decks.keys()),
        pending_plans=missing_plans, comparator=config['comparator'],
        comparator_sha256=sha(config['comparator']),
        pending_reason='Remaining whole forecasts and approved exact contexts; nested ozone initialization/time and radiation effective-radius consumers remain native.',
        backend_scope='cloudiness hex-cuda; every other slot native-cpu',
        missing_fields_are_not_numerical_differences=True, private_ratchet_created=False)
    temporary = args.out.with_suffix('.partial')
    temporary.write_text(json.dumps(out, indent=2) + '\n')
    temporary.replace(args.out)
    print(json.dumps(dict(bound=len(decks), complete_forecasts=sum(
        row['status'].startswith('complete forecast') for row in decks.values()),
        accepted=sum(row['accepted'] for row in decks.values()), required=required['required_requests'])))


if __name__ == '__main__':
    main()
