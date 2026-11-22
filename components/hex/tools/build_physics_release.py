#!/usr/bin/env python3
"""Build and seal physics assets inside the pinned publish.yml manylinux image.

Run with an empty output/cache and the release image's Python/GNU/Rust tools.
No credentials, upload or publication are part of this recipe.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

from woof.hex.mpasphys.native_library_build import referee_builder
nlb = referee_builder()
from woof.hex.mpasphys import resources_build as rb
from woof.hex.mpasphys.resources_fetch import SCHEMA

IMAGE = 'quay.io/pypa/manylinux_2_28_x86_64@sha256:4131ddd7acb9c6c966af02fe7e71edc51e078c2d41f475966697676430bfa2ae'
SYSTEM = {'libc.so.6','libm.so.6','libdl.so.2','libpthread.so.0','librt.so.1',
          'libresolv.so.2','libutil.so.1','ld-linux-x86-64.so.2'}


def run(*command):
    return subprocess.check_output(command, text=True)


def graft(library):
    """Keep the complete non-glibc DT_NEEDED closure beside each entry point."""
    destination = library.parent/'lib'
    destination.mkdir(exist_ok=True)
    dependencies = {}
    for line in run('ldd', str(library)).splitlines():
        match = re.match(r'\s*(\S+) => (/\S+) ', line)
        if match and match[1] not in SYSTEM:
            dependencies[match[1]] = Path(match[2]).resolve()
        elif 'not found' in line:
            raise RuntimeError(f'unresolved dependency: {line}')
    for name, source in dependencies.items():
        if source != (destination/name).resolve():
            shutil.copyfile(source, destination/name)
        if run('patchelf','--print-rpath',str(destination/name)).strip()!='$ORIGIN':
            subprocess.run(['patchelf','--set-rpath','$ORIGIN',str(destination/name)],check=True)
        (destination/name).chmod(0o755)
    if run('patchelf','--print-rpath',str(library)).strip()!='$ORIGIN/lib':
        subprocess.run(['patchelf','--set-rpath','$ORIGIN/lib',str(library)],check=True)
    return {name:{'source':str(path),'sha256':rb.sha256_file(destination/name)}
            for name,path in sorted(dependencies.items())}


def audit_elf(path, runtime_paths):
    versions = run('readelf','--version-info',str(path))
    required = sorted({tuple(map(int, value.split('.'))) for value in re.findall(r'\bGLIBC_([0-9.]+)', versions)})
    if required and required[-1] > (2,28):
        raise RuntimeError(f'{path}: exceeds the glibc 2.28 release policy: {required[-1]}')
    dependencies = {}
    for line in run('ldd',str(path)).splitlines():
        match = re.match(r'\s*(\S+) => (/\S+) ', line)
        if match:
            dependency=Path(match[2]).resolve()
            if match[1] not in SYSTEM and dependency not in runtime_paths:
                raise RuntimeError(f'{path}: dependency escapes the release closure: {line}')
            dependencies[match[1]]=str(dependency)
        elif 'not found' in line:
            raise RuntimeError(f'{path}: unresolved dependency: {line}')
    return {'glibc_max':'.'.join(map(str,required[-1])) if required else None,
            'dynamic':run('readelf','--dynamic',str(path)),'dependencies':dependencies}


def host_elf(path):
    with path.open('rb') as stream:
        header=stream.read(20)
    # CUBINs are ELF too, but are CUDA images rather than host executables.
    return header[:4]==b'\x7fELF' and int.from_bytes(header[18:20],'little')==62


def repin_packet(path, cache, changed):
    """Re-pin relocated ELF helpers without changing the table-word pins."""
    packet=rb.load_packet(path)
    def update(value):
        if isinstance(value,dict):
            source=value.get('path')
            if isinstance(source,str) and Path(source) in changed and 'sha256' in value:
                value['sha256']=changed[Path(source)]
            for child in value.values(): update(child)
        elif isinstance(value,list):
            for child in value: update(child)
    update(packet)
    text=json.dumps(rb.portable_packet(packet,cache),sort_keys=True,indent=2)+'\n'
    digest=hashlib.sha256(text.encode()).hexdigest()
    target=cache/'packets'/f'physics-resources-{digest}.json'
    target.write_text(text)
    return {'path':str(target.relative_to(cache)), 'sha256':digest, 'bytes':target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--phase', choices=('native','packets','audit','seal'), default='native')
    a = parser.parse_args()
    if run('getconf','GNU_LIBC_VERSION').strip() != 'glibc 2.28':
        raise RuntimeError('build physics assets only inside the pinned manylinux_2_28 release image')
    a.cache = a.cache.resolve(); a.out = a.out.resolve()
    a.cache.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    receipt_path = a.out/f'{a.phase}-build-receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'image':IMAGE,'started_utc':datetime.now(timezone.utc).isoformat(),'stages':{},
        'tools':{tool:run(tool,'--version').splitlines()[0] for tool in ('gcc','gfortran','patchelf')},
        'numerics_note':'native frameworks and strict Thompson tables are shared; CUDA packet images differ by fast/strict and architecture',
    }
    def stage(name, function):
        start=time.perf_counter()
        result={'status':'running'}
        try:
            value=function()
        except BaseException as error:
            result.update(status='failed',error=type(error).__name__,detail=str(error))
            raise
        else:
            result['status']='done'
            return value
        finally:
            receipt['stages'][name]={**result,'wall_seconds':time.perf_counter()-start,
                                    'finished_utc':datetime.now(timezone.utc).isoformat()}
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    if a.phase == 'native':
        mpi=stage('mpich',lambda:nlb.build_mpich(a.cache,rb.fetch_url,jobs=a.jobs,offline=False,log=rb._log))
        stage('pnetcdf',lambda:nlb.build_pnetcdf(a.cache,rb.fetch_url,mpi,jobs=a.jobs,offline=False,log=rb._log))
        def native(pin):
            return nlb.build_native_library(a.cache,fetch=rb.fetch_url,fetch_file=rb.fetch,mpich=mpi,
                        jobs=max(2,a.jobs//2),offline=False,log=rb._log,pin_set=pin,static_gnu_runtime=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pin:pool.submit(native,pin) for pin in rb.PIN_SETS}
            rows={rb.ROUTE_KEYS[pin]['native-library']:future.result() for pin,future in futures.items()}
        for key,row in rows.items():
            if row['licence']['compiled'] or 'noahmp' not in row['licence']['stubbed']:
                raise RuntimeError(f'{key}: default assets must contain the Noah-MP licence stub')
        rb.record_installed(a.cache,rows)
        receipt['native_resources']=rows
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        return

    # Native and packet phases are independent and can use the same bounded
    # release container concurrently. Sealing never recompiles either.
    rows=rb.read_installed(a.cache)['resources']
    if a.phase == 'packets':
        import woof
        receipt['engine_source']=str(Path(woof.__file__).resolve())
        rows={}
        rows['pool-reader']=stage('pool-reader',lambda:rb.stage_pool_reader(a.cache))
        rows['mpas-source']=stage('mpas-source',lambda:rb.assemble_mpas_source(a.cache))
        rows['mpas-source-rrfs']=stage('rrfs-source',lambda:rb.assemble_rrfs_source(a.cache))
        variants={}
        for mode in ('fast','strict'):
            variants[mode]={}
            for arch in ('sm_89','sm_90','sm_100','sm_120'):
                # Each process binds one numerics mode. All common native/table data
                # are reused by their content pins, without compiler substitutions.
                script=('import json; from pathlib import Path; '
                        'from woof.hex.mpasphys.resources_build import build_packet; '
                        f'print(json.dumps(build_packet(Path({str(a.cache)!r}),image_arch={arch!r})))')
                env=dict(os.environ,WOOF_HEX_NUMERICS=mode)
                def packet():
                    value=subprocess.check_output([sys.executable,'-c',script],text=True,env=env)
                    result=json.loads(value)
                    if 'bl_mynnedmf' in result['not_built'] or 'mp_thompson_aerosols' in result['not_built']:
                        raise RuntimeError(f"incomplete release packet: {result['not_built']}")
                    return result
                result=stage(f'packet-{mode}-{arch}',packet)
                path=Path(result['packet'])
                variants[mode][arch]={'path':str(path.relative_to(a.cache)),'sha256':result['packet_sha256'],
                                      'bytes':path.stat().st_size}
        rows['physics-resources']={**variants['fast']['sm_89'],'variants':variants}
        rows['thompson-tables']={'path':str(a.cache/'thompson-strict/tables'),
                                'sha256':hashlib.sha256(json.dumps(rb._thompson_expected(),sort_keys=True).encode()).hexdigest()}
        rb.record_installed(a.cache,rows)
        receipt['resources']=rows
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        return
    variants=rows['physics-resources']['variants']
    for key in ('native-library','native-library-rrfs'):
        library=Path(rows[key]['path'])
        grafted=stage('graft-'+key,lambda:graft(library))
        record=json.loads(library.with_name('build-receipt.json').read_text())
        record.setdefault('pre_graft_library_sha256',record['library_sha256'])
        record['library_sha256']=rb.sha256_file(library)
        record['release_graft']=grafted
        library.with_name('build-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        rows[key]['sha256']=record['library_sha256']
    members={'common':set(),'mpas':set(),'rrfs':set()}
    def collect(path, group):
        path=Path(path)
        if path.is_dir():
            for child in path.rglob('*'):
                if child.is_file(): collect(child,group)
        elif path.is_file():
            path.relative_to(a.cache) # Refuse external paths, including forgotten build dependencies.
            members[group].add(path)
    for key,row in rows.items():
        group='rrfs' if key.endswith('-rrfs') else 'mpas' if key in ('native-library','mpas-source') else 'common'
        path=Path(row['path'])
        if not path.is_absolute(): path=a.cache/path
        collect(path,group)
        if key.startswith('native-library'):
            collect(path.parent,group)
    for arches in variants.values():
        for item in arches.values():
            path=a.cache/item['path']; collect(path,'common')
            def discover(value):
                if isinstance(value,dict):
                    for child in value.values(): discover(child)
                elif isinstance(value,list):
                    for child in value: discover(child)
                elif isinstance(value,str) and Path(value).is_absolute():
                    collect(value,'common')
            discover(rb.load_packet(path))
    # Copy the complete notices with every asset set, once in the common asset.
    notices=a.cache/'release-licenses';notices.mkdir(exist_ok=True)
    tree=Path(__file__).resolve().parents[1]
    for path in [tree/'LICENSE',tree/'NOTICE',tree/'THIRD-PARTY-NOTICES.txt',*sorted((tree/'licenses').glob('*')),
                 *sorted((tree/'src/hexcore/data/licenses').glob('*'))]:
        if path.is_file(): shutil.copyfile(path,notices/path.name)
    for pattern in ('native/mpich-*/install/share/doc/mpich/*','native/pnetcdf-*/install/share/doc/*'):
        for path in a.cache.glob(pattern):
            if path.is_file(): shutil.copyfile(path,notices/path.name)
    # Source builders retire their unpacked trees. Read the notices directly
    # from their SHA-verified input archives, including embedded MPI components.
    for component in (nlb.MPICH,nlb.PNETCDF):
        archive=a.cache/'native/downloads'/component['sha256']/component['url'].rsplit('/',1)[1]
        if rb.sha256_file(archive) != component['sha256']:
            raise RuntimeError(f'{archive}: source archive differs from its pin')
        with tarfile.open(archive,'r:gz') as source:
            for member in source:
                if member.isfile() and re.match(r'^(COPYING|COPYRIGHT|LICENSE|LICENCE)([._-].*)?$',Path(member.name).name,re.I):
                    relative=Path(member.name)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise RuntimeError(f'unsafe licence member: {member.name}')
                    target=notices/'dependencies'/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with source.extractfile(member) as src,target.open('wb') as dst:
                        shutil.copyfileobj(src,dst)
    # GNU runtime notices apply even when libgfortran/libgcc are linked statically.
    runtime_notices=notices/'gcc-runtime';runtime_notices.mkdir(exist_ok=True)
    for path in sorted(Path('/usr/share/licenses/libgcc').glob('COPYING*')):
        shutil.copyfile(path,runtime_notices/path.name)
    quadmath_notice=Path('/usr/share/licenses/libquadmath/COPYING.LIB.libquadmath')
    if quadmath_notice.is_file():
        shutil.copyfile(quadmath_notice,runtime_notices/quadmath_notice.name)
    if not (runtime_notices/'COPYING.RUNTIME').is_file():
        raise RuntimeError('release image is missing the GCC runtime exception notice')
    (runtime_notices/'README.txt').write_text(
        'Native MPAS frameworks: MPAS BSD notice. GCC runtime closure: libgfortran, libgcc and '
        'libatomic under GPL with GCC Runtime Library Exception. If present, libquadmath '
        'is under LGPL-2.1; its full notice is beside these files. The release owner records '
        'the remaining project licence question in the readiness report. Noah-MP is stubbed.\n')
    pciaccess_notice=Path('/usr/share/licenses/libpciaccess/COPYING')
    if pciaccess_notice.is_file():
        shutil.copyfile(pciaccess_notice,notices/'libpciaccess-COPYING')
    collect(notices,'common')
    receipt['helper_grafts']={}
    changed={}
    # Rust helpers can name libgcc_s even with static NetCDF. Keep their
    # closure as portable as the native entry points, without host symlinks.
    for group,paths in members.items():
        for path in sorted(paths.copy()):
            if host_elf(path) and path.parent.name!='lib' and not path.name.startswith('libmpasphys_native'):
                receipt['helper_grafts'][str(path.relative_to(a.cache))]=graft(path)
                changed[path]=rb.sha256_file(path)
                collect(path.parent/'lib',group)
    rows['pool-reader']['sha256']=rb.sha256_file(Path(rows['pool-reader']['path']))
    # patchelf changes helper bytes, so packets must pin the relocated binaries,
    # not the preceding build copies. Native table and decoded-word pins stay fixed.
    for arches in variants.values():
        for arch,item in arches.items():
            old=a.cache/item['path']
            replacement=repin_packet(old,a.cache,changed)
            members['common'].discard(old)
            collect(a.cache/replacement['path'],'common')
            arches[arch]=replacement
    rows['physics-resources']={**variants['fast']['sm_89'],'variants':variants}
    # Packet table paths also occur in the native source trees. Give every
    # member one archive owner: fetch refuses overlaps rather than racing two
    # workers that create the same destination with exclusive writes.
    shared = members['common'] | (members['mpas'] & members['rrfs'])
    members['common'].update(shared)
    for group in ('mpas', 'rrfs'):
        members[group].difference_update(members['common'])
    decoded=stage('relocated-decoder-words',lambda:rb.decoded_digests(
        rb.load_packet(a.cache/variants['fast']['sm_89']['path'])))
    if any(rb.DECODED_SHA256.get(key,value)!=value for key,value in decoded.items()):
        raise RuntimeError(f'relocated helper changed certified table words: {decoded}')
    receipt['relocated_decoded_sha256']=decoded
    receipt['elf_audit']={}
    runtime_paths={path.resolve() for paths in members.values() for path in paths}
    for group,paths in members.items():
        for path in sorted(paths):
            if host_elf(path):
                receipt['elf_audit'][str(path.relative_to(a.cache))]=audit_elf(path,runtime_paths)
    rb.record_installed(a.cache,rows)
    # MPAS-Data is fetched from its upstream pin, never redistributed in a ZIP.
    table_pins={}
    for pin in rb.PIN_SETS:
        for name,row in nlb.source_manifest(pin)['tables'].items():
            if row['origin']=='mpas-data':
                old=table_pins.setdefault(row['sha256'], (name,row))
                if old!=(name,row):
                    raise RuntimeError(f'inconsistent MPAS-Data pin: {name}')
    upstream={}
    for paths in members.values():
        for path in paths.copy():
            digest=rb.sha256_file(path)
            if digest in table_pins:
                name,row=table_pins[digest]
                entry=upstream.setdefault(digest, {
                    'name':name,'bytes':row['bytes'],'sha256':digest,
                    'url':rb.URLS['mpas-data'].format(path=row['path']),
                    'origin':'MPAS-Dev/MPAS-Data','commit':rb.MPAS_DATA_COMMIT,
                    'pin_sets':list(rb.PIN_SETS),'paths':[]})
                entry['paths'].append(str(path.relative_to(a.cache)))
                paths.remove(path)
    for row in upstream.values():
        row['paths'].sort()
    receipt['upstream_files']=list(upstream.values())
    receipt['runtime_closure']={key:sorted(value['dependencies'])
                              for key,value in receipt['elf_audit'].items()
                              if Path(key).name.startswith('libmpasphys_native')}
    receipt['licence_readiness_question']='libquadmath LGPL-2.1 if present; release owner handles readiness ruling'
    receipt['distribution_holds']=[]
    receipt['resources']=rows
    receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    if a.phase=='audit':
        print(json.dumps({'receipt':str(receipt_path),'runtime_closure':receipt['runtime_closure']}))
        return
    assets=[]
    def archive_group(pair):
        group,paths=pair
        filename=f'physics-resources-{group}-manylinux_2_28_x86_64.zip'
        target=a.out/filename
        inventory={}
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for path in sorted(paths):
                name=str(path.relative_to(a.cache))
                archive.write(path,name)
                inventory[name]={'sha256':rb.sha256_file(path),'bytes':path.stat().st_size,
                                 'executable':bool(path.stat().st_mode & 0o111)}
        return {'name':filename,'bytes':target.stat().st_size,'sha256':rb.sha256_file(target),
                       'pin_sets':list(rb.PIN_SETS) if group=='common' else ['v8.4.2' if group=='mpas' else 'rrfs'],
                       'files':inventory}
    with ThreadPoolExecutor(max_workers=3) as pool:
        assets=list(pool.map(archive_group,members.items()))
    resources={}
    for key,row in rows.items():
        value=dict(row);path=Path(value['path'])
        value['path']=str(path.relative_to(a.cache)) if path.is_absolute() else str(path)
        value['pin_sets']=['rrfs'] if key.endswith('-rrfs') else ['v8.4.2'] if key in ('native-library','mpas-source') else list(rb.PIN_SETS)
        resources[key]=value
    pins={'schema':SCHEMA,'image':IMAGE,'platform':'manylinux_2_28_x86_64',
          'release_assets':{'repository':'recastsystems/gpuwm-hex','tag':'physics-resources-20261017'},
          'resources':resources,'assets':assets,'upstream_files':list(upstream.values()),
          'licence':{'noahmp':'stubbed; UCAR code is not compiled or shipped',
                     'gnu_runtime':{'attempt_flags':['-static-libgfortran','-static-libgcc'],
                                    'dynamic_closure':receipt['runtime_closure'],
                                    'notices':'release-licenses/gcc-runtime',
                                    'quadmath':'LGPL-2.1 when present; readiness question belongs to the release owner'},
                     'native':{k:v['licence'] for k,v in rows.items() if k.startswith('native-library')}}}
    a.manifest.write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
    receipt['assets']=[{k:v for k,v in asset.items() if k!='files'} for asset in assets]
    receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
