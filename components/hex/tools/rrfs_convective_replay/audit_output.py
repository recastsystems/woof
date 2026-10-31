"""Whole-run output audit launcher; instrumentation times are not speed evidence.

OUTPUT_AUDIT names the JSON receipt. Count raw staging calls by thread, actual
bridge searches, and stream-view construction per output event. Run the real CLI.
"""
import json
import os
from pathlib import Path
import sys
import threading
from collections import Counter
from time import perf_counter

counts = Counter()
searches = []
events = []


def profile(frame, event, arg):
    if (event == 'c_call' and getattr(arg, '__name__', '') == 'tofile'
            and frame.f_code.co_name == 'write_stream'
            and frame.f_code.co_filename.endswith(('restart_v842.py', 'restart_rrfs.py'))):
        counts[threading.current_thread().name] += 1


def install():
    from woof.hex import engines
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics as D
    original, views, before = engines._resolve, D.views, D.before_write
    def resolve(spec, *args, **kwargs):
        t = perf_counter()
        result = original(spec, *args, **kwargs)
        if spec.name == 'rw_mpas_stream':
            searches.append(dict(thread=threading.current_thread().name,
                                 ms=1000*(perf_counter()-t)))
        return result
    engines._resolve = resolve
    def event_views(self):
        if self._event_views is None:
            self._audit_view_builds = getattr(self, '_audit_view_builds', 0)+1
        return views(self)
    def before_write(self, **kw):
        old = getattr(self, '_audit_view_builds', 0)
        result = before(self, **kw)
        events.append(dict(step=kw['step'], views_built=self._audit_view_builds-old))
        return result
    D.views, D.before_write = event_views, before_write
    sys.setprofile(profile)
    threading.setprofile(profile)


if __name__ == '__main__':
    install()
    try:
        from woof.hex import cli
        a = cli.build_parser().parse_args(sys.argv[1:])
        cli._select_numerics(a)
        code = int(a.handler(a))
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        Path(os.environ['OUTPUT_AUDIT']).write_text(json.dumps(dict(
            tofile_calls_by_thread=dict(counts), bridge_searches=searches,
            output_events=events, timing='instrumented smoke, not speed evidence'), indent=2)+'\n')
    sys.exit(code)
