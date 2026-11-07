"""Verified canonical vortex candidate to the shared Rust WPS exporter.

This is an engine input door, with no assimilation implementation. The caller
supplies the original soil-layer declaration. All required surface fields must
already exist in the canonical candidate; the door invents no surface donors.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time


def export(*, request_path, composition_path, valid_time, output, workers, surface_policy="complete"):
    from woof.mapped_engine_bridge import require_engine
    from woof.ingest.preparation_workers import worker_environment
    import woof
    started=time.perf_counter()
    if surface_policy not in ('complete','native-gfs-skin'):
        raise ValueError('unsupported TC WPS surface policy')
    if not isinstance(workers,int) or workers<1:
        raise ValueError('TC export needs a positive worker count')
    request_path=Path(request_path).resolve(strict=True)
    raw=request_path.read_bytes();request=json.loads(raw)
    if set(request)!={'schema','before','after','library','before_sha256','after_sha256','library_sha256'} or request['schema']!='gpuwm-tc-source-init/1':
        raise ValueError('invalid TC initialization declaration')
    def path(name):
        p=Path(request[name])
        return (request_path.parent/p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True)
    before,after,library=(path(n) for n in ('before','after','library'))
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    for name,p in (('before',before/'frames.json'),('after',after/'frames.json'),('library',library)):
        if digest(p)!=request[name+'_sha256']:raise ValueError(f'TC {name} identity changed')
    transform=json.loads((after/'tc-transform.json').read_text())
    declaration=transform['declaration']
    patch=json.loads((after/'tc-patch-receipt.json').read_text())
    transform_digest=hashlib.sha256(json.dumps(declaration,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if (declaration['mode']!='relocate' or declaration['valid_time']!=valid_time
        or declaration['background_manifest_sha256']!=request['before_sha256']
        or transform['output_manifest_sha256']!=request['after_sha256']
        or patch['background_manifest_sha256']!=request['before_sha256']
        or patch['transform_sha256']!=transform_digest):
        raise ValueError('TC export clock or transform provenance differs')
    # WPS pressure/surface/soil fields have a closed contract. A carried species
    # needs its own native handoff before this door can preserve it.
    if declaration['carried_fields']:
        raise ValueError('TC WPS export cannot preserve additional carried species')
    composition_path=Path(composition_path).resolve(strict=True)
    contract=Path(woof.__file__).with_name('authorities')/'wps-intermediate-fields.v1.json'
    output=Path(output).absolute()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('TC WPS output must be new, in an existing directory')
    if surface_policy=='native-gfs-skin':
        # Native MPAS initial GFS dispatch consumes SKINTEMP. Separate SST is
        # an optional surface-update source, not an initial atmospheric field.
        # Snow and sea ice remain required from this same canonical source.
        doc=json.loads(contract.read_text())
        doc['bindings']=[row for row in doc['bindings'] if row['wps']!='SST']
        contract=output.with_name(output.name+'.field-contract.json')
        if contract.exists():raise ValueError('TC WPS field contract already exists')
        contract.write_text(json.dumps(doc,indent=2)+'\n')
    pending=output.with_name(output.name+'.part')
    if pending.exists():raise ValueError('TC WPS pending output already exists')
    composition_digest=digest(composition_path)
    argv=[str(require_engine()),'wps','--frameset',str(after),'--composition',str(composition_path),
          '--contract',str(contract),'--valid-time',valid_time,'--out',str(pending),
          '--longitude-origin','zero']
    completed=subprocess.run(argv,capture_output=True,text=True,env=worker_environment(workers))
    if completed.returncode:
        raise RuntimeError('Rust TC WPS export failed: '+completed.stderr[-4000:])
    native=json.loads(completed.stdout.splitlines()[-1])
    if (native.get('longitude_origin_deg') != 0
        or native['frameset_sha256']!=request['after_sha256']
        or native['composition_sha256']!=composition_digest
        or native['valid_time']!=valid_time):
        raise ValueError('TC WPS export receipt differs from admitted inputs')
    pending.replace(output)
    native['output']['path']=str(output)
    receipt={'schema':'gpuwm-tc-wps-export/1','mode':'relocate',
             'request_sha256':hashlib.sha256(raw).hexdigest(),
             'transform_sha256':transform_digest,'declaration':declaration,
             'output':str(output),'surface_policy':surface_policy,'export':native,'wall_seconds':time.perf_counter()-started,
             'qualification':'requires native initialization and forecast verification'}
    output.with_name(output.name+'.receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return output,receipt
