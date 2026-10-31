#!/usr/bin/env python3
"""Verify geography archives and unpack them through native tar.

Python handles paths, hashes and native process receipts. Archive decompression
and extraction run in tar; weather fields are never decoded or rewritten here.
All dataset paths, checksums and storage bounds are explicit request data.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_archive(row, tar):
    path = Path(row['path']).resolve(strict=True)
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('geography archive differs from pinned bytes: ' + str(path))
    argv = [str(tar), '--list', '--verbose', '--numeric-owner',
            '--quoting-style=escape', '--file', str(path)]
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    entries, size, roots = 0, 0, set()
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or parts[0][0] not in ('-', 'd'):
            raise ValueError('unsupported geography archive member: ' + line)
        name = parts[5]
        member = Path(name)
        if (member.is_absolute() or '..' in member.parts or '\\' in name
                or not member.parts):
            raise ValueError('unsafe geography archive member: ' + name)
        entries += 1
        size += int(parts[2])
        roots.add(member.parts[0])
    if not entries:
        raise ValueError('empty geography archive')
    return dict(row, path=str(path), entries=entries, extracted_bytes=size,
                roots=sorted(roots), listing_argv=argv,
                listing_sha256=hashlib.sha256(result.stdout.encode()).hexdigest())


def preflight(request, tar):
    if not request['archives']:
        raise ValueError('explicit geography archives required')
    archives = [inspect_archive(row, tar) for row in request['archives']]
    if len({row['path'] for row in archives}) != len(archives):
        raise ValueError('duplicate geography archive')
    return dict(schema='mpas.rrfs-native-geography-archive/v1', archives=archives,
                extracted_bytes=sum(row['extracted_bytes'] for row in archives),
                tar_sha256=sha(tar), completed=True)


def sparse_member():
    """Launch native sparse byte copying for one already inspected tar member."""
    root=Path(os.environ['MPAS_GEOGRAPHY_OUT']).resolve(strict=True)
    member=Path(os.environ['TAR_FILENAME'])
    if member.is_absolute() or '..' in member.parts or not member.parts:
        raise ValueError('unsafe sparse geography member')
    counter=root/'.allocated-bytes'
    allocated=int(counter.read_text())
    budget=int(os.environ['MPAS_GEOGRAPHY_BUDGET'])
    if allocated+int(os.environ['TAR_SIZE'])>budget:
        raise ValueError('next geography member exceeds the reserved storage bound')
    target=root/member
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise ValueError('duplicate sparse geography member')
    subprocess.run([os.environ['MPAS_GEOGRAPHY_CP'],'--sparse=always','/dev/stdin',str(target)],check=True)
    if target.stat().st_size!=int(os.environ['TAR_SIZE']):
        raise ValueError('native sparse geography copy length differs')
    counter.write_text(str(allocated+target.stat().st_blocks*512)+'\n')


def compare_member():
    """Compare an original native tar member with its read-only filesystem view."""
    root = Path(os.environ['MPAS_GEOGRAPHY_VIEW']).resolve(strict=True)
    member = Path(os.environ['TAR_FILENAME'])
    if member.is_absolute() or '..' in member.parts or not member.parts:
        raise ValueError('unsafe geography comparison member')
    target = root / member
    if not target.is_file() or target.stat().st_size != int(os.environ['TAR_SIZE']):
        raise ValueError('compressed geography member length differs')
    subprocess.run([os.environ['MPAS_GEOGRAPHY_CMP'], '-', str(target)], check=True)


@contextmanager
def compressed_view(index, tar, out, work_root, mount_root, limit_bytes,
                    sqfstar, squashfuse, library_directory, reserve_bytes=0,
                    preflight_index_sha256=None):
    """Expose complete archive bytes without allocating their expanded size.

    Native decompressors and sqfstar produce images inside the bounded work
    directory. Read-only virtual mounts use the caller's TMPDIR; no expanded
    file is stored there. Native cmp verifies every original regular member.
    All mounts are released on success or failure of the consuming native run.
    """
    import shlex
    work_root = work_root.resolve(strict=True)
    temporary_root = Path(os.environ['TMPDIR']).resolve(strict=True)
    out, mount_root = out.resolve(), mount_root.resolve()
    if (not out.is_relative_to(work_root) or out == work_root or out.exists()
            or not mount_root.is_relative_to(temporary_root)
            or mount_root == temporary_root or mount_root.exists()):
        raise ValueError('compressed geography requires new owned work and TMPDIR paths')
    if not index['completed'] or sha(tar) != index['tar_sha256']:
        raise ValueError('geography source or native tar changed')
    if preflight_index_sha256 is None:
        if preflight(dict(archives=index['archives']), tar) != index:
            raise ValueError('geography source or native tar changed')
    else:
        # A pinned complete preflight already inspected every member through
        # this exact tar binary. Authenticate that receipt and every archive
        # byte; repeating decompression solely to list unchanged bytes adds no
        # source evidence. Unpinned callers still perform a fresh preflight.
        encoded = (json.dumps(index, indent=2) + '\n').encode()
        if hashlib.sha256(encoded).hexdigest() != preflight_index_sha256:
            raise ValueError('pinned geography preflight receipt changed')
        for row in index['archives']:
            path = Path(row['path']).resolve(strict=True)
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('geography archive differs from pinned bytes: ' + str(path))
    usage = int(subprocess.check_output(['du', '-s', '-B1', str(work_root)], text=True).split()[0])
    if limit_bytes <= 0 or reserve_bytes < 0 or usage + reserve_bytes >= limit_bytes:
        raise ValueError('compressed geography exceeds the explicit work-directory storage bound')
    # sqfstar dispatches by argv[0] and may be a symlink to mksquashfs.
    sqfstar, squashfuse = sqfstar.absolute(), squashfuse.absolute()
    if not sqfstar.is_file() or not squashfuse.is_file():
        raise ValueError('native compressed filesystem tools are absent')
    cmp = Path(shutil.which('cmp')).resolve(strict=True)
    unmount = Path(shutil.which('fusermount3')).resolve(strict=True)
    environment = dict(os.environ, LD_LIBRARY_PATH=str(library_directory.resolve(strict=True))
                       + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    out.mkdir(parents=True)
    mount_root.mkdir(parents=True)
    mounted, products, allocated = [], [], 0
    try:
        for number, row in enumerate(index['archives']):
            source = Path(row['path'])
            if source.name.endswith('.tar.bz2'):
                decompressor = Path(shutil.which('bzip2')).resolve(strict=True)
            elif source.name.endswith('.tar.gz'):
                decompressor = Path(shutil.which('gzip')).resolve(strict=True)
            else:
                raise ValueError('explicit native compressed tar format required')
            image = out / (row['sha256'] + '.sqfs')
            argv = [str(sqfstar), str(image), '-processors', '4', '-mem', '512M',
                    '-comp', 'zstd', '-no-progress']
            with (out / (str(number) + '-compression.log')).open('w') as log:
                decode = subprocess.Popen([str(decompressor), '-dc', str(source)],
                                          stdout=subprocess.PIPE, stderr=log)
                compress = subprocess.Popen(argv, stdin=decode.stdout, stdout=log, stderr=log)
                try:
                    while compress.poll() is None:
                        size = image.stat().st_blocks * 512 if image.exists() else 0
                        if usage + allocated + size + reserve_bytes >= limit_bytes:
                            raise ValueError('compressed geography reaches the reserved storage bound')
                        time.sleep(0.1)
                    if compress.returncode != 0:
                        raise ValueError('native geography compression failed')
                    # Tar consumers may stop at the end marker before the
                    # decompressor emits trailing padding. Drain through cat
                    # so the original decompressor has a witnessed zero exit.
                    subprocess.run(['cat'], stdin=decode.stdout,
                                   stdout=subprocess.DEVNULL, check=True)
                    if decode.wait() != 0:
                        raise ValueError('native geography decompression failed')
                finally:
                    decode.stdout.close()
                    for process in (compress, decode):
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
            allocated += image.stat().st_blocks * 512
            if usage + allocated + reserve_bytes >= limit_bytes:
                raise ValueError('compressed geography exceeds the reserved storage bound')
            view = mount_root / row['sha256']
            view.mkdir()
            subprocess.run([str(squashfuse), '-o', 'ro', str(image), str(view)],
                           env=environment, check=True)
            mounted.append(view)
            if not os.statvfs(view).f_flag & os.ST_RDONLY:
                raise ValueError('geography filesystem view is not read-only')
            member_command = shlex.join([sys.executable, str(Path(__file__).resolve()),
                                         '_compare-member'])
            compare_environment = dict(environment, MPAS_GEOGRAPHY_VIEW=str(view),
                                       MPAS_GEOGRAPHY_CMP=str(cmp))
            compare_directory = out / (str(number) + '-member-check')
            compare_directory.mkdir()
            subprocess.run([str(tar), '--extract', '--file', str(source),
                            '--directory', str(compare_directory), '--no-same-owner',
                            '--to-command=' + member_command], env=compare_environment,
                           check=True)
            shutil.rmtree(compare_directory)
            for name in row['roots']:
                target = out / name
                if target.exists() or target.is_symlink():
                    raise ValueError('duplicate geography dataset root')
                target.symlink_to(view / name, target_is_directory=True)
            products.append(dict(source_sha256=row['sha256'], image=image.name,
                                 image_sha256=sha(image), bytes=image.stat().st_size,
                                 allocated_bytes=image.stat().st_blocks * 512,
                                 compression_argv=argv, decompressor_sha256=sha(decompressor),
                                 all_regular_members_native_cmp_identical=True))
        indexes = {}
        for row in index['archives']:
            for name in row['roots']:
                for path in sorted((out / name).rglob('index')):
                    indexes[str(path.relative_to(out))] = sha(path)
        after = int(subprocess.check_output(['du', '-s', '-B1', str(work_root)], text=True).split()[0])
        if after + reserve_bytes >= limit_bytes:
            raise ValueError('complete compressed geography exceeds the work-directory storage bound')
        result = dict(index, storage='native-read-only-compressed-filesystem',
                      work_bytes_before=usage, work_bytes_after=after,
                      work_limit_bytes=limit_bytes, reserved_bytes=reserve_bytes,
                      sqfstar_sha256=sha(sqfstar), squashfuse_sha256=sha(squashfuse),
                      cmp_sha256=sha(cmp), products=products, indexes=indexes)
        (out / 'MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
        yield result
    finally:
        for view in reversed(mounted):
            subprocess.run([str(unmount), '-u', str(view)], check=True)


@contextmanager
def reuse_compressed_view(manifest, manifest_sha256, images, out, work_root,
                         mount_root, limit_bytes, squashfuse, library_directory,
                         reserve_bytes=0):
    """Mount previously byte-verified native images after authenticating them.

    The pinned original receipt includes cmp of every source tar member. Image
    and archive hashes are checked again before reuse; no weather bytes change.
    """
    work_root = work_root.resolve(strict=True)
    temporary_root = Path(os.environ['TMPDIR']).resolve(strict=True)
    images, out, mount_root = images.resolve(strict=True), out.resolve(), mount_root.resolve()
    if (not images.is_relative_to(work_root) or images == work_root
            or not out.is_relative_to(work_root) or out == work_root or out.exists()
            or not mount_root.is_relative_to(temporary_root)
            or mount_root == temporary_root or mount_root.exists()):
        raise ValueError('reused geography requires owned work and new TMPDIR paths')
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    if hashlib.sha256(encoded).hexdigest() != manifest_sha256:
        raise ValueError('verified geography receipt changed')
    if (not manifest['completed']
            or manifest['storage'] != 'native-read-only-compressed-filesystem'
            or sha(squashfuse) != manifest['squashfuse_sha256']):
        raise ValueError('verified geography or native mount tool changed')
    products = {p['source_sha256']:p for p in manifest['products']}
    if (len(products) != len(manifest['products'])
            or set(products) != {r['sha256'] for r in manifest['archives']}):
        raise ValueError('verified geography image inventory changed')
    for row in manifest['archives']:
        source, product = Path(row['path']), products[row['sha256']]
        image = images / product['image']
        if (Path(product['image']).name != product['image'] or image.is_symlink()
                or not product['all_regular_members_native_cmp_identical']
                or source.stat().st_size != row['bytes'] or sha(source) != row['sha256']
                or image.stat().st_size != product['bytes'] or sha(image) != product['image_sha256']):
            raise ValueError('verified geography source or image bytes changed')
    usage = int(subprocess.check_output(['du','-s','-B1',str(work_root)],text=True).split()[0])
    if limit_bytes <= 0 or reserve_bytes < 0 or usage + reserve_bytes >= limit_bytes:
        raise ValueError('reused geography exceeds the explicit work-directory storage bound')
    environment = dict(os.environ,LD_LIBRARY_PATH=str(library_directory.resolve(strict=True))
                       + ':' + os.environ.get('LD_LIBRARY_PATH',''))
    unmount = Path(shutil.which('fusermount3')).resolve(strict=True)
    out.mkdir(parents=True)
    mount_root.mkdir(parents=True)
    mounted = []
    try:
        for row in manifest['archives']:
            view = mount_root / row['sha256']; view.mkdir()
            subprocess.run([str(squashfuse),'-o','ro',str(images/products[row['sha256']]['image']),str(view)],
                           env=environment,check=True)
            mounted.append(view)
            if not os.statvfs(view).f_flag & os.ST_RDONLY:
                raise ValueError('reused geography view is not read-only')
            for name in row['roots']:
                target = out / name
                if target.exists() or target.is_symlink():
                    raise ValueError('duplicate geography dataset root')
                target.symlink_to(view/name,target_is_directory=True)
        for name,digest in manifest['indexes'].items():
            if sha(out/name) != digest:
                raise ValueError('reused geography index changed')
        result = dict(manifest,reused_image_manifest_sha256=manifest_sha256,
                      image_directory=str(images),work_bytes_current=usage)
        (out/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
        yield result
    finally:
        for view in reversed(mounted):
            subprocess.run([str(unmount),'-u',str(view)],check=True)


def add_pinned_datasets(view, datasets):
    """Expose additional original datasets after authenticating every member.

    Native GWD may require geography that differs from the land-use selection
    used by the static interpolation. Dataset names and member inventories are
    explicit request data; no field is decoded or rewritten here.
    """
    view = view.resolve(strict=True)
    manifest = view / 'MANIFEST.json'
    original_sha256 = sha(manifest)
    original = json.loads(manifest.read_text())
    checked = {}
    for name, request in datasets.items():
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('simple supplemental geography dataset name required')
        target = view / name
        if target.exists() or target.is_symlink():
            raise ValueError('supplemental geography cannot replace an existing dataset')
        source = Path(request['source']).resolve(strict=True)
        expected = request['files']
        actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
        if not expected or actual != set(expected) or any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('supplemental geography member inventory differs')
        for member, record in expected.items():
            if Path(member).is_absolute() or '..' in Path(member).parts:
                raise ValueError('unsafe supplemental geography member')
            path = source / member
            if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError('supplemental geography bytes changed')
        if not request.get('authority') or not request.get('source_receipt_sha256'):
            raise ValueError('supplemental geography requires original source authority')
        checked[name] = dict(request, source=str(source))
    # All sources are authenticated before any path or receipt is changed.
    for name, request in checked.items():
        (view / name).symlink_to(request['source'], target_is_directory=True)
    result = dict(original, base_view_manifest_sha256=original_sha256,
                  supplemental_datasets=checked)
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    return result


def extract(index, tar, out, work_root, limit_bytes, sparse=False, reserve_bytes=0):
    work_root = work_root.resolve(strict=True)
    out = out.resolve()
    if not out.is_relative_to(work_root) or out == work_root or out.exists():
        raise ValueError('geography extraction requires a new directory inside the work root')
    if not index['completed'] or sha(tar) != index['tar_sha256']:
        raise ValueError('geography preflight or native tar changed')
    checked = preflight(dict(archives=index['archives']), tar)
    if checked != index:
        raise ValueError('geography archive listing changed after preflight')
    usage = int(subprocess.check_output(['du', '-s', '-B1', str(work_root)], text=True).split()[0])
    if limit_bytes <= 0 or reserve_bytes < 0 or usage+reserve_bytes>=limit_bytes:
        raise ValueError('geography extraction exceeds the explicit work-directory storage bound')
    if not sparse and usage + index['extracted_bytes'] + reserve_bytes > limit_bytes:
        raise ValueError('geography extraction exceeds the explicit work-directory storage bound')
    out.mkdir(parents=True, exist_ok=False)
    cp=Path(shutil.which('cp')).resolve(strict=True)
    if sparse:
        (out/'.allocated-bytes').write_text('0\n')
        env=dict(os.environ,MPAS_GEOGRAPHY_OUT=str(out),MPAS_GEOGRAPHY_BUDGET=str(limit_bytes-usage-reserve_bytes),MPAS_GEOGRAPHY_CP=str(cp))
        import shlex
        member_command=shlex.join([sys.executable,str(Path(__file__).resolve()),'_sparse-member'])
    for row in index['archives']:
        argv=[str(tar),'--extract','--file',row['path'],'--directory',str(out),'--no-same-owner']
        if sparse:
            subprocess.run([*argv,'--to-command='+member_command],env=env,check=True)
        else:
            subprocess.run(argv,check=True)
    manifest = dict(index, work_bytes_before=usage, work_limit_bytes=limit_bytes,
                    reserved_bytes=reserve_bytes,sparse_native_copy=sparse,
                    cp_sha256=sha(cp) if sparse else None,
                    allocated_file_bytes=int((out/'.allocated-bytes').read_text()) if sparse else None,
                    indexes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('index'))})
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    check = sub.add_parser('preflight')
    check.add_argument('--request', type=Path, required=True)
    check.add_argument('--out', type=Path, required=True)
    unpack = sub.add_parser('extract')
    for name in ('index', 'out', 'work-root'):
        unpack.add_argument('--' + name, type=Path, required=True)
    unpack.add_argument('--limit-bytes', type=int, required=True)
    unpack.add_argument('--sparse',action='store_true')
    unpack.add_argument('--reserve-bytes',type=int,default=0)
    sub.add_parser('_sparse-member')
    sub.add_parser('_compare-member')
    args = parser.parse_args()
    if args.action=='_sparse-member':
        sparse_member()
        sys.exit(0)
    if args.action == '_compare-member':
        compare_member()
        sys.exit(0)
    tar = Path(shutil.which('tar')).resolve(strict=True)
    if args.action == 'preflight':
        result = preflight(json.loads(args.request.read_text()), tar)
        with args.out.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
    else:
        result = extract(json.loads(args.index.read_text()), tar, args.out, args.work_root, args.limit_bytes,args.sparse,args.reserve_bytes)
    print(json.dumps(dict(completed=result['completed'], extracted_bytes=result['extracted_bytes'])))
