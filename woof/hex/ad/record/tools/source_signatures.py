"""Identify simple CUDA parameter layouts from image-matched saved sources.

Only a small, explicit grammar is recognized. Unknown or ambiguous types
remain unresolved. A CArray contains a device pointer at byte zero followed
by size, shape and stride integers; CIndexer contains only integers. The
matching CuPy carray.cuh must be retained with the observation.
"""
# SPDX-License-Identifier: Apache-2.0
import hashlib
import gzip
import json
import bisect
import re
import subprocess
import zlib
from pathlib import Path

FUNCTION = re.compile(r'extern\s+"C"\s+__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s+)?void\s+(\w+)\s*\((.*?)\)\s*\{', re.S)
ARRAY = re.compile(r'^(?:const\s+)?CArray<\s*[^,<>]+\s*,\s*(\d+)\s*,\s*[01]\s*,\s*[01]\s*>\s+\w+$')
INDEXER = re.compile(r'^(?:const\s+)?CIndexer<\s*(\d+)\s*,\s*[01]\s*>\s+\w+$')
# Pointer depth does not change the kernel argument ABI: it is one device
# address.  Whether the pointed-to memory contains further addresses is a
# separate relocation question and cannot be inferred from this width alone.
POINTER = re.compile(r'^(?:const\s+)?(?:unsigned\s+)?(?:char|short|int|int[234]|long|long long|float|double|bool|void|uint32_t|uint64_t|int32_t|int64_t)(?:\s*\*\s*(?:const\b\s*)?)+[A-Za-z_]\w*$')
SCALAR = re.compile(r'^(?:const\s+)?(unsigned int|unsigned long long|unsigned|long long|uint32_t|uint64_t|int32_t|int64_t|int|float|double|bool)\s+\w+$')
WIDTH = {'bool': 1, 'int': 4, 'unsigned': 4, 'unsigned int': 4, 'uint32_t': 4, 'int32_t': 4,
         'float': 4, 'long long': 8, 'unsigned long long': 8,
         'uint64_t': 8, 'int64_t': 8, 'double': 8}

# A comma or parenthesis in a CUDA comment is not part of a declaration.
# Preserve quoted text while removing comments, including C++ raw strings;
# otherwise a literal containing // or /* could erase later declarations.
COMMENT_OR_LITERAL = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^()\s\\]{0,16})\([\s\S]*?\)(?P=delimiter)")'
    r'|(?P<string>"(?:\\.|[^"\\])*")'
    r"|(?P<char>'(?:\\.|[^'\\])*')"
    r'|(?P<line>//[^\r\n]*)|(?P<block>/\*[\s\S]*?\*/)')


def remove_comments(text):
    def replace(match):
        if match.lastgroup in ('line', 'block'):
            return re.sub(r'[^\r\n]', ' ', match.group())
        return match.group()
    return COMMENT_OR_LITERAL.sub(replace, text)


def split_parameters(text):
    result, begin, depth = [], 0, 0
    for i, char in enumerate(text):
        if char == '<':
            depth += 1
        elif char == '>':
            depth -= 1
            if depth < 0:
                return None
        elif char == ',' and depth == 0:
            result.append(' '.join(text[begin:i].split()))
            begin = i + 1
    if depth:
        return None
    result.append(' '.join(text[begin:].split()))
    return [] if result == [''] else result


def signature(text):
    parameters = split_parameters(remove_comments(text))
    if parameters is None:
        return None
    result = []
    for parameter in parameters:
        # Restrict is an aliasing promise on the pointed-to object; it does not
        # change the kernel's pointer width. Unknown qualifiers still fail.
        parameter = ' '.join(re.sub(r'\b(?:__restrict__|__restrict)\b', ' ', parameter).split())
        match = ARRAY.fullmatch(parameter)
        if match:
            result.append(('carray', 16 + 16 * int(match[1])))
            continue
        match = INDEXER.fullmatch(parameter)
        if match:
            result.append(('indexer', 8 + 16 * int(match[1])))
            continue
        if POINTER.fullmatch(parameter):
            result.append(('device_pointer', 8))
            continue
        match = SCALAR.fullmatch(parameter)
        if match:
            result.append(('scalar', WIDTH[match[1]]))
            continue
        return None
    return result


def image_signatures(source):
    """Get explicit signatures and conservatively expand known source macros.

    Only these two image-matched, include-free translation units require macro
    expansion. A signature is accepted from expanded text only if it is the
    same with the strict and fast launch-bounds switches. The preprocessor is
    barred from loading headers or using non-task source files.
    """
    content = remove_comments(source.decode('utf-8'))
    result = {}
    for name, parameters in FUNCTION.findall(content):
        parsed = signature(parameters)
        if name in result and result[name] != parsed:
            result[name] = None
        elif name not in result:
            result[name] = parsed
    if not any(x in content for x in ('DECLARE_EDGE_KERNEL', 'W_FILTER_BOUNDS')):
        return result
    if re.search(r'^\s*#\s*include\b', content, re.M):
        return result
    variants = []
    for fast in (False, True):
        command = ['cc', '-E', '-P', '-nostdinc', '-undef', '-x', 'c']
        if fast:
            command += ['-DHEX_NUMERICS_FAST=1']
        command += ['-']
        output = subprocess.run(command, input=content, text=True, capture_output=True,
                                check=True, timeout=15).stdout
        expanded = {}
        for name, parameters in FUNCTION.findall(output):
            parsed = signature(parameters)
            if name in expanded and expanded[name] != parsed:
                expanded[name] = None
            elif name not in expanded:
                expanded[name] = parsed
        variants.append(expanded)
    for name, parsed in variants[0].items():
        if parsed is not None and variants[1].get(name) == parsed:
            if name in result and result[name] not in (None, parsed):
                result[name] = None
            else:
                result[name] = parsed
    return result


def from_cache(cache):
    """Map (driver image sha256, kernel name) to an unambiguous layout.

    CuPy cache files in this observation prefix the image with a 40-byte SHA1
    text digest. The driver observer saves the exact image plus one NUL byte.
    This mapping must be checked against the observer's module SHA-256 values.
    """
    signatures, sources = {}, {}
    for source in Path(cache).glob('*.cubin.cu'):
        binary = source.with_suffix('')
        if not binary.is_file():
            continue
        raw = binary.read_bytes()
        if (len(raw) < 44 or not re.fullmatch(b'[0-9a-f]{40}', raw[:40]) or
                raw[40:44] != b'\x7fELF' or hashlib.sha1(raw[40:]).hexdigest().encode() != raw[:40]):
            continue
        image_sha = hashlib.sha256(raw[40:] + b'\0').hexdigest()
        source_bytes = source.read_bytes()
        sources[image_sha] = {'source_sha256': hashlib.sha256(source_bytes).hexdigest(), 'path': str(source)}
        for name, parsed in image_signatures(source_bytes).items():
            key = (image_sha, name)
            if key in signatures and signatures[key] != parsed:
                signatures[key] = None
            elif key not in signatures:
                signatures[key] = parsed
    return signatures, sources


def _complete_json_rows(path):
    """Read flushed JSON lines, preserving the fact a gzip may be unfinished."""
    decoder = zlib.decompressobj(31)
    pending, result = b'', []
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1 << 20):
            pending += decoder.decompress(chunk)
            lines = pending.split(b'\n')
            pending = lines.pop()
            result.extend(json.loads(line) for line in lines if line)
    return result, decoder.eof


def from_observation(observer, driver_images):
    """Resolve source layouts through observed module loads and exact images.

    The observer associates the source passed to CuPy's compiler with the
    returned module, including cache hits. Require the preceding observed load,
    both content hashes and byte equality with the separately recorded driver
    image before using a layout. An interrupted stream provides only its
    complete flushed rows, and is never considered a completed recording.
    """
    observer, driver_images = Path(observer), Path(driver_images)
    bindings, bindings_closed = _complete_json_rows(observer / 'source-bindings.jsonl.gz')
    events, events_closed = _complete_json_rows(observer / 'modules.jsonl.gz')
    loads = {}
    for event in events:
        if event.get('action') == 'load':
            loads.setdefault(event.get('module'), []).append(event)
    for module in loads:
        loads[module].sort(key=lambda row: row['clock_ns'])
    signatures, sources = {}, {}
    stats = {'bindings': len(bindings), 'accepted': 0, 'no_loaded_event': 0,
             'no_driver_image': 0, 'bindings_stream_closed': bindings_closed,
             'module_stream_closed': events_closed}
    checked_images, checked_sources = {}, {}
    for row in bindings:
        raw_sha, source_sha = row.get('image_sha256'), row.get('source_sha256')
        if (row.get('schema') != 'ad.binder.phys-source-binding/v1' or row.get('complete') is not False or
                not isinstance(row.get('module'), int) or row['module'] <= 0 or
                not isinstance(row.get('clock_ns'), int) or
                not isinstance(raw_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', raw_sha) or
                not isinstance(source_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', source_sha)):
            raise ValueError('invalid observed source binding')
        candidates = loads.get(row['module'], [])
        index = bisect.bisect_right([event['clock_ns'] for event in candidates], row['clock_ns']) - 1
        if index < 0 or candidates[index].get('image_sha256') != raw_sha:
            stats['no_loaded_event'] += 1
            continue
        if raw_sha not in checked_images:
            image = gzip.decompress((observer / 'images' / (raw_sha + '.gz')).read_bytes())
            if hashlib.sha256(image).hexdigest() != raw_sha:
                raise ValueError('observed module image SHA-256 mismatch')
            driver_bytes = image + b'\0'
            driver_sha = hashlib.sha256(driver_bytes).hexdigest()
            driver_path = driver_images / (driver_sha + '.gz')
            if not driver_path.is_file():
                checked_images[raw_sha] = None
            else:
                if gzip.decompress(driver_path.read_bytes()) != driver_bytes:
                    raise ValueError('driver module image is not byte-identical')
                checked_images[raw_sha] = driver_sha
        image_sha = checked_images[raw_sha]
        if image_sha is None:
            stats['no_driver_image'] += 1
            continue
        source_path = observer / 'sources' / (source_sha + '.cu.gz')
        if source_sha not in checked_sources:
            content = gzip.decompress(source_path.read_bytes())
            if hashlib.sha256(content).hexdigest() != source_sha:
                raise ValueError('observed compiler source SHA-256 mismatch')
            checked_sources[source_sha] = content
        content = checked_sources[source_sha]
        if len(content) != row.get('source_bytes'):
            raise ValueError('observed compiler source size mismatch')
        source_entry = sources.setdefault(image_sha, {'source_sha256s': [], 'observation': str(observer)})
        if source_sha not in source_entry['source_sha256s']:
            source_entry['source_sha256s'].append(source_sha)
        for name, parsed in image_signatures(content).items():
            key = (image_sha, name)
            if key in signatures and signatures[key] != parsed:
                signatures[key] = None
            elif key not in signatures:
                signatures[key] = parsed
        stats['accepted'] += 1
    return signatures, sources, stats


def _capture_object(root, record, suffix):
    """Read one content-addressed capture object, without following an escape."""
    if not isinstance(record, dict):
        raise ValueError('missing capture object record')
    sha, name, length = record.get('sha256'), record.get('path'), record.get('bytes')
    if (not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha) or
            name != f'objects/{sha}.{suffix}' or not isinstance(length, int) or length < 0):
        raise ValueError('invalid capture object identity')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('capture object escapes its archive')
    content = path.read_bytes()
    if len(content) != length or hashlib.sha256(content).hexdigest() != sha:
        raise ValueError('capture object length or SHA-256 mismatch')
    return content


def _capture_abi(arguments):
    """Use the capture contract's types, never its recorded argument values."""
    if not isinstance(arguments, list):
        return None
    result = []
    for argument in arguments:
        if not isinstance(argument, dict):
            return None
        kind, depth, dtype = argument.get('kind'), argument.get('pointer_depth'), argument.get('dtype')
        if kind == 'device' and isinstance(depth, int) and depth > 0:
            result.append(('device_pointer', 8))
        elif kind == 'scalar' and depth == 0 and isinstance(dtype, str):
            match = re.fullmatch(r'[<|]([iufb])(1|2|4|8)', dtype)
            if match is None or (match[1] == 'b' and match[2] != '1'):
                return None
            result.append(('scalar', int(match[2])))
        else:
            return None
    return result


def from_capture_archive(observer, driver_images, archive, config):
    """Bind capture ABI contracts to exact images seen in this observation.

    The manifest identifies the archived source and image, not live launch
    values, allocations, ownership, branches or scheduling. Only a manifest
    whose CuPy image equals this recording's image byte-for-byte (and whose
    driver image equals it with the observed trailing NUL) can contribute.
    The resulting layouts are still insufficient to construct a replay.
    """
    observer, driver_images, archive = Path(observer), Path(driver_images), Path(archive)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', config) or config in ('.', '..'):
        raise ValueError('invalid capture configuration')
    launches, launches_closed = _complete_json_rows(observer / 'launches.jsonl.gz')
    pairs = set()
    for row in launches:
        name, image = row.get('kernel'), row.get('image_sha256')
        if (isinstance(name, str) and re.fullmatch(r'[A-Za-z_]\w*', name) and
                'stochastic' not in name.lower() and isinstance(image, str) and
                re.fullmatch(r'[0-9a-f]{64}', image)):
            pairs.add((name, image))
    signatures, sources, records, rejected = {}, {}, [], []
    stats = {'observed_pairs': len(pairs), 'launch_stream_closed': launches_closed,
             'accepted': 0, 'no_capture': 0, 'image_not_recorded': 0,
             'incomplete_manifests': 0, 'unsupported_abi': 0, 'source_abi_conflict': 0}
    images, objects, source_layouts = {}, {}, {}
    config_root = (archive / config).resolve(strict=True)
    if not config_root.is_relative_to(archive.resolve()):
        raise ValueError('capture configuration escapes its archive')
    for name, image_sha in sorted(pairs):
        if image_sha not in images:
            image_path = observer / 'images' / (image_sha + '.gz')
            if not image_path.is_file():
                images[image_sha] = None
            else:
                image = gzip.decompress(image_path.read_bytes())
                if hashlib.sha256(image).hexdigest() != image_sha:
                    raise ValueError('same-run CuPy image SHA-256 mismatch')
                driver_sha = hashlib.sha256(image + b'\0').hexdigest()
                driver_path = driver_images / (driver_sha + '.gz')
                if not driver_path.is_file():
                    images[image_sha] = None
                else:
                    if gzip.decompress(driver_path.read_bytes()) != image + b'\0':
                        raise ValueError('same-run driver image mismatch')
                    images[image_sha] = (image, driver_sha)
        observed = images[image_sha]
        if observed is None:
            stats['image_not_recorded'] += 1
            rejected.append({'kernel': name, 'image_sha256': image_sha, 'reason': 'image_not_recorded'})
            continue
        kernel_dir = (config_root / name).resolve()
        if not kernel_dir.is_relative_to(config_root) or not kernel_dir.is_dir():
            stats['no_capture'] += 1
            rejected.append({'kernel': name, 'image_sha256': image_sha, 'reason': 'no_capture'})
            continue
        selected = None
        # Capture first launches across every recorded step. This bounded
        # search makes no claim about images appearing only in later launches.
        steps = sorted((p for p in kernel_dir.iterdir() if p.is_dir() and p.name.isdecimal()),
                       key=lambda p: int(p.name))
        for step in steps:
            path = step / 'launch-000000' / 'manifest.json'
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(config_root):
                raise ValueError('capture manifest escapes its configuration')
            content = path.read_bytes()
            manifest = json.loads(content)
            if (manifest.get('schema') != 'ad.hex.launch/v1' or type(manifest.get('complete')) is not bool or
                    manifest.get('config') != config or manifest.get('kernel') != name):
                raise ValueError('invalid capture launch manifest')
            if manifest['complete'] is False:
                stats['incomplete_manifests'] += 1
                continue
            if manifest.get('image', {}).get('sha256') == image_sha:
                selected = (path, content, manifest)
                break
        if selected is None:
            stats['no_capture'] += 1
            rejected.append({'kernel': name, 'image_sha256': image_sha, 'reason': 'no_capture'})
            continue
        path, manifest_bytes, manifest = selected
        for field, suffix in (('image', 'cubin'), ('source', 'cu')):
            record = manifest.get(field)
            key = (field, record.get('sha256') if isinstance(record, dict) else None)
            if (not isinstance(record, dict) or not isinstance(key[1], str) or
                    not re.fullmatch(r'[0-9a-f]{64}', key[1]) or
                    record.get('path') != f'objects/{key[1]}.{suffix}' or
                    not isinstance(record.get('bytes'), int) or record['bytes'] < 0):
                raise ValueError('invalid capture object identity')
            if key not in objects:
                objects[key] = _capture_object(archive, record, suffix)
            elif len(objects[key]) != record.get('bytes'):
                raise ValueError('inconsistent capture object length')
        if objects[('image', image_sha)] != observed[0]:
            raise ValueError('capture and same-run module images differ')
        layout = _capture_abi(manifest.get('arguments'))
        if layout is None:
            stats['unsupported_abi'] += 1
            rejected.append({'kernel': name, 'image_sha256': image_sha, 'reason': 'unsupported_abi',
                             'manifest': str(path), 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()})
            continue
        source_sha = manifest['source']['sha256']
        if source_sha not in source_layouts:
            source_layouts[source_sha] = image_signatures(objects[('source', source_sha)])
        declared = source_layouts[source_sha].get(name)
        if declared is not None and declared != layout:
            stats['source_abi_conflict'] += 1
            rejected.append({'kernel': name, 'image_sha256': image_sha, 'reason': 'source_abi_conflict',
                             'driver_image_sha256': observed[1],
                             'manifest': str(path), 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                             'source_sha256': source_sha, 'capture_layout': layout, 'source_layout': declared})
            continue
        driver_sha = observed[1]
        key = (driver_sha, name)
        if key in signatures and signatures[key] != layout:
            signatures[key] = None
        elif key not in signatures:
            signatures[key] = layout
        record = {'kernel': name, 'image_sha256': image_sha, 'driver_image_sha256': driver_sha,
                  'source_sha256': source_sha, 'manifest': str(path),
                  'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                  'arguments': len(layout)}
        records.append(record)
        sources.setdefault(driver_sha, []).append(record)
        stats['accepted'] += 1
    return signatures, sources, {'schema': 'ad.binder.phys-capture-abi-audit/v1',
                                 'complete': False, **stats, 'records': records, 'rejected': rejected}


def add_verified(signatures, sources, proof_path, source_path):
    """Add a source that passed verify_diagnostics_source's exact-image check."""
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('exact_match_including_elf_terminator') is not True:
        raise ValueError('source verification did not match the recorded image')
    source_bytes = gzip.decompress(Path(source_path).read_bytes())
    if hashlib.sha256(source_bytes).hexdigest() != proof['source_sha256']:
        raise ValueError('verified source SHA-256 mismatch')
    image_sha = proof['recorded_image_sha256']
    sources[image_sha] = {'source_sha256': proof['source_sha256'], 'path': str(source_path),
                          'proof': str(proof_path)}
    for name, parsed in image_signatures(source_bytes).items():
        key = (image_sha, name)
        if key in signatures and signatures[key] != parsed:
            signatures[key] = None
        elif key not in signatures:
            signatures[key] = parsed


def from_bundle(bundle):
    """Read the saved source bundle and authenticate each source's bytes."""
    root = Path(bundle)
    index = json.loads((root / 'index.json').read_text())
    signatures = {}
    for image, record in index['images'].items():
        content = gzip.decompress((root / record['path']).read_bytes())
        if hashlib.sha256(content).hexdigest() != record['source_sha256']:
            raise ValueError(f'saved source SHA-256 mismatch for {image}')
        for name, parsed in image_signatures(content).items():
            key = (image, name)
            if key in signatures and signatures[key] != parsed:
                signatures[key] = None
            elif key not in signatures:
                signatures[key] = parsed
    return signatures
