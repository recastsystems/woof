"""Collect actual p25 native-pool ABI/CUDA receipts; never forecast acceptance.

Each operation is executed by test_native_surface_service_v842.py against the
strict f3 library. Missing, skipped, stale or nonzero output cannot pass.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re


def collect(directory, receipt_directory=None, job_path=None, pipeline_exit=None, summary_path=None):
    root = Path(directory).resolve()
    source = root / 'hex/tree/src/hexcore/mpasphys/surface_update_v842.py'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    unit = re.search(r'^SERVICE_UNIT = "([^"]+)"', source.read_text(), re.M).group(1)
    job_path = job_path or root / 'receipts/native-service-jobs-v7.json'
    job = json.loads(job_path.read_text())
    receipt_root = receipt_directory or root / 'receipts/native-service'
    operations = ('physics_update_surface', 'physics_update_sst', 'physics_update_sstskin',
                  'physics_update_deepsoiltemp', 'update_radiation_diagnostics', 'update_convection_step2')
    errors, records = [], []
    for operation in operations:
        path = receipt_root / (operation + '.json')
        if not path.is_file():
            errors.append('missing actual operation receipt: ' + operation)
            continue
        row = json.loads(path.read_text())
        service = row.get('service', {})
        counts = service.get('operations', {})
        if (row.get('process_count') != 1 or row.get('operation') != operation
                or service.get('module_sha256') != source_sha or service.get('executed') is not True
                or service.get('unit') != unit or service.get('backend') != 'hex-cuda'
                or counts.get(operation) != 1 or set(counts) != set(operations) or sum(counts.values()) != 1
                or row.get('native_library_sha256') != job['native_library_sha256']
                or row.get('cuda_compiler') != 'direct NVRTC'
                or row.get('cuda_options') != job['cuda_options']
                or not isinstance(row.get('words'), int) or row['words'] <= 0
                or row.get('differing_words') != 0 or row.get('halo_differing_words') != 0
                or row.get('failures')):
            errors.append('actual operation/provenance/word comparison failed: ' + operation)
        records.append(dict(operation=operation, words=row.get('words'),
                            differing_words=row.get('differing_words'),
                            halo_differing_words=row.get('halo_differing_words'),
                            path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    exit_path = pipeline_exit or root / 'receipts/native-service-gpu.exit'
    if not exit_path.is_file() or exit_path.read_text().strip() != '0':
        errors.append('actual slot-backed pipeline exit is absent or nonzero')
    log = receipt_root / 'run.log'
    if not log.is_file() or not re.search(r'\b8 passed\b', log.read_text()) or 'skipped' in log.read_text():
        errors.append('all eight service tests must execute and pass without skips')
    summary = dict(schema='p25-native-service-replay/v1', process_count=1,
                   tier='subsystem native-pool ABI/CUDA only', forecast_identity_claimed=False,
                   module_sha256=source_sha, native_library_sha256=job['native_library_sha256'],
                   cuda_compiler='direct NVRTC', cuda_options=job['cuda_options'],
                   cases=len(records), words=sum(r['words'] or 0 for r in records),
                   differing_words=sum(r['differing_words'] or 0 for r in records),
                   halo_differing_words=sum(r['halo_differing_words'] or 0 for r in records),
                   passed=not errors, errors=errors, records=records)
    destination = summary_path or root / 'receipts/native-service-replay-summary.json'
    destination.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--receipt-directory', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--pipeline-exit', type=Path)
    parser.add_argument('--summary', type=Path)
    arguments = parser.parse_args()
    result = collect(arguments.directory, arguments.receipt_directory, arguments.job,
                     arguments.pipeline_exit, arguments.summary)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
