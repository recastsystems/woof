"""Preserve closed native stream bytes on the archive host before retiring copies.

MPAS mpas_atm_core.F:760,827,862 at fork 1d6e370e writes a stream before
the next Begin timestep banner. A later clock is required before transfer;
the final collection happens only after native exit. Inputs stay protected.
"""
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime,timedelta

spec = importlib.util.spec_from_file_location('rrfs_archive_products',
                                            Path(__file__).parent.parent/'verify_products.py')
products = importlib.util.module_from_spec(spec); spec.loader.exec_module(products)
sha = products.sha


def matches(template, name):
    pattern = re.escape(template)
    for macro in ('Y', 'M', 'D', 'h', 'm', 's'):
        pattern = pattern.replace(re.escape('$'+macro), r'\d{4}' if macro == 'Y' else r'\d{2}')
    return re.fullmatch(pattern, name) is not None


def readonly_inputs(streams, names):
    """Only explicitly selected pure input streams may retain linked ingress."""
    streams = list(streams)
    for name in names:
        uses = [s for s in streams if matches(s.get('filename_template', ''), name)]
        if not uses or any(s.get('type') != 'input' for s in uses):
            raise ValueError('read-only ingress is not exclusively a native input stream: '+name)
    return set(names)


def expected_outputs(runtime, streams, inactive):
    ordinary = [s for s in streams if not s.get('output_timelevels')]
    expected = products.expected_forecast(runtime, ordinary, inactive) if any(
        'output' in s.get('type','') for s in ordinary) else {}
    start = datetime.strptime(runtime['start'], '%Y-%m-%d_%H:%M:%S')
    duration = products.interval_seconds(runtime['duration'])
    for stream in streams:
        if 'output' not in stream.get('type','') or not stream.get('output_timelevels'):
            continue
        packages = stream.get('packages','').split(';') if stream.get('packages') else []
        if packages and all(q in inactive for q in packages):
            continue
        for hour in map(int, stream.get('output_timelevels').split()):
            if hour < 0 or hour*3600 > duration:
                continue
            clock = (start+timedelta(hours=hour)).strftime('%Y-%m-%d_%H:%M:%S')
            expected.setdefault(products.filename(stream.get('filename_template'),clock),[]).append(clock)
    return expected


class ArchiveOutputs:
    def __init__(self, directory, runtime, request, protected):
        self.directory = directory
        self.request = request
        self.protected = set(protected)
        self.files = {}
        self.comparator = Path(request['comparator']).resolve(strict=True)
        self.expected = expected_outputs(runtime,
            ET.parse(directory/'streams.atmosphere').getroot(), request['inactive_packages'])
        self.stage = Path(request['stage'])
        if not self.stage.is_absolute() or '..' in self.stage.parts:
            raise ValueError('absolute owned archive staging path required')
        command = 'test ! -e '+shlex.quote(str(self.stage))+' && mkdir -p -- '+shlex.quote(str(self.stage))
        subprocess.run(['ssh','-o','BatchMode=yes',request['host'],command],check=True)

    def collect(self, finished=False):
        log = self.directory/'log.atmosphere.0000.out'
        clocks = re.findall(r'Begin timestep (\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)',
                            log.read_text()) if log.exists() else []
        current = clocks[-1] if clocks else None
        for file in sorted(self.directory.glob('*.nc')):
            if file.is_symlink() or file.name in self.protected or file.name not in self.expected:
                continue
            if file.name in self.files:
                if sha(file) != self.files[file.name]['sha256']:
                    raise ValueError('native rewrote an already archived stream')
                continue
            expected = self.expected[file.name]
            if not finished and (not current or current <= max(expected)):
                continue
            before = file.stat(); digest = sha(file)
            inventory = json.loads(subprocess.check_output(
                [str(self.comparator),'--field-digests',str(file)],text=True))
            if not inventory['variables'] or any('word_sha256' not in q for q in inventory['variables']):
                raise ValueError('closed native stream lacks full raw field digests')
            shapes = [q['shape'][q['dimensions'].index('Time')] for q in inventory['variables'] if 'Time' in q['dimensions']]
            if inventory['xtime'] != expected or not shapes or any(n != len(expected) for n in shapes):
                raise ValueError('closed native stream records differ from stream clocks')
            remote = self.stage/file.name
            subprocess.run(['rsync','-a','--',str(file),self.request['host']+':'+str(remote)],check=True)
            command = shlex.join(['sha256sum','--',str(remote)])
            remote_digest = subprocess.check_output(['ssh','-o','BatchMode=yes',self.request['host'],command],text=True).split()[0]
            after = file.stat()
            if remote_digest != digest or sha(file) != digest or (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('closed stream changed during byte-preserving transfer')
            self.files[file.name] = dict(sha256=digest,bytes=before.st_size,
                archive_sha256=remote_digest,expected_clocks=expected,inventory=inventory,
                closure_clock=current,closure_after_native_exit=finished,
                private_copy_retired=bool(self.request['retire_private']))
            self.write()
            if self.request['retire_private']:
                file.unlink()
        if self.request.get('work_root'):
            usage = int(subprocess.check_output(['du','-s','-B1',self.request['work_root']],text=True).split()[0])
            if usage > self.request['limit_bytes']:
                raise ValueError('native output buffers exceeded the owned scratch bound')

    def write(self):
        witness = dict(schema='mpas.native-archived-output-bytes/v1',
            request=self.request,comparator_sha256=sha(self.comparator),files=self.files,
            no_field_transform=True,original_bytes_retained_on_archive_host=True)
        path = self.directory/'ARCHIVED-OUTPUTS.json'
        temporary=path.with_suffix('.partial')
        temporary.write_text(json.dumps(witness,indent=2)+'\n');temporary.replace(path)
