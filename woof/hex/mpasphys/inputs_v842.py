"""Stage shared native tables and project timed Rust surface records by rank."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import time

import numpy as np


def stage_tables(runtime, arguments, layout, config, out, row):
    """Run the suite's actual selectors once before MPI workers or CUDA exist."""
    began = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    roots = runtime.native_table_roots(layout.source, config,
                                       getattr(arguments, 'native_physics_tables', None))
    selector = row.get('native_table_selector')
    if selector is None:
        tables = (table for root in roots for table in sorted(root.iterdir())
            if table.is_file() and (table.suffix in ('.TBL', '.DBL') or
                                   table.name.startswith(('RRTMG_', 'CCN_'))))
    else:
        module, name = selector.split(':', 1)
        callback = getattr(import_module(module), name)
        keyword = row.get('native_table_override_keyword')
        tables = callback(roots[:1], **{keyword: roots[1:]}) if keyword else callback(roots)
    staged = []
    for table in tables:
        table = Path(table)
        target = Path(out) / table.name
        if target.exists() and target.resolve() != table.resolve():
            raise ValueError(f'native table destination already differs: {target}')
        if not target.exists():
            target.symlink_to(table.resolve())
        staged.append(table.name)
    initialization = None
    selector = row.get('native_initialization_namelist')
    if selector is not None:
        module, name = selector.split(':', 1)
        # The run's namelist file, or the one it resolved from --physics-suite (forecast_v842.run_forecast).
        staged_namelist = (getattr(arguments, 'physics_namelist', None)
                           or getattr(arguments, 'resolved_physics_namelist', None))
        if staged_namelist is None:
            raise ValueError('the route stages the run namelist into the native working directory, and this run '
                             'has neither --physics-namelist nor a resolved suite namelist')
        initialization = getattr(import_module(module), name)(staged_namelist, out)
    return dict(files=staged, initialization_namelist_sha256=initialization,
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-began, parallelism='one host staging process; before CUDA')


class HostField:
    """Rust record installer ABI over host storage, without importing CUDA."""

    def __init__(self, size):
        self.words = np.empty(size, np.float32)

    def get(self):
        return self.words.copy()

    def set(self, words):
        self.words[...] = words


class ProjectedSurfaceRecordReader:
    """Decode once on rank zero, scatter native global cell words to each rank."""

    def __init__(self, comm, partitions, factory, library, path):
        self.comm, self.partitions = comm, partitions
        self.rank = comm.Get_rank()
        self.reader = (factory(library, path, n_cells=partitions[0]._global_cells)
                       if self.rank == 0 else None)
        self.times = comm.bcast(None if self.reader is None else self.reader.times, root=0)
        self.last_record = None
        self.reads = []

    def __call__(self, current_date, pool):
        began = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        dates = self.comm.allgather(current_date)
        if len(set(dates)) != 1:
            raise ValueError('native surface clocks differ between ranks')
        packets, failure = None, None
        if self.rank == 0:
            try:
                global_pool = {name: HostField(self.partitions[0]._global_cells)
                               for name in ('sst', 'xice')}
                self.reader(current_date, global_pool)
                def packet(part):
                    return self.reader.last_record, {
                        name: np.ascontiguousarray(field.words[part.cell_l2g])
                        for name, field in global_pool.items()}
                with ThreadPoolExecutor(max_workers=len(self.partitions)) as workers:
                    packets = list(workers.map(packet, self.partitions))
            except Exception as error:
                failure = f'{type(error).__name__}: {error}'
        failure = self.comm.bcast(failure, root=0)
        if failure is not None:
            raise ValueError(f'native surface record projection failed: {failure}')
        record, words = self.comm.scatter(packets, root=0)
        # Validate both fields before publishing either, retaining the local
        # allocation-only dummy and all non-surface pool fields untouched.
        targets = {}
        extent = len(self.partitions[self.rank].cell_l2g)
        for name in ('sst', 'xice'):
            host = pool[name].get()
            if (host.dtype != np.float32 or host.ndim != 1 or host.size not in (extent, extent+1)
                    or words[name].dtype != np.float32 or words[name].shape != (extent,)):
                raise ValueError(f'native surface {name} rank extent or dtype differs')
            host[:extent] = words[name]
            targets[name] = host
        for name, host in targets.items():
            pool[name].set(host)
        self.last_record = record
        self.reads.append(dict(record=record, started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.perf_counter()-began,
            parallelism=f'{len(self.partitions)} MPI ranks; Rust decode once, parallel projection'))
