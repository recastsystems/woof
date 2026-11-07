"""Card-free observation FDDA allocation inventory, including the report book.

Rust counts native decoded levels. Admission may subsequently discard rows;
pricing the decoded count is a stated upper bound, never a guessed multiplier.
"""
from functools import lru_cache
from pathlib import Path
import subprocess


def _sources(cfg):
    path=Path(cfg.obs_inputfile or f'OBS_DOMAIN{cfg.obs_domain_id}01').resolve()
    files=[path]
    if path.name.startswith('OBS_DOMAIN') and path.name.endswith('01'):
        for volume in range(2,100):
            following=path.with_name(path.name[:-2]+f'{volume:02}')
            if not following.is_file():
                break
            files.append(following)
    return tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files)


@lru_cache(maxsize=64)
def _count(sources, binary):
    result=subprocess.run([binary,'--count-rows',sources[0][0]],check=True,
                          capture_output=True,text=True)
    return int(result.stdout.strip())


def array_shapes(cfg):
    if not cfg.obs_nudge_opt:
        return {}
    from woof import bridges
    binary=bridges.find_artifact('GPUWM_RW_FDDAOBS',bridges.executable_name('rw_fddaobs'),
                                crate_relative=bridges.RUSTWX_CRATE_RELATIVE)
    if binary is None:
        raise FileNotFoundError('FDDAOBS_RUST_READER: observation memory admission requires rw_fddaobs')
    count=_count(_sources(cfg),str(binary))
    nxy=(cfg.nx+1)*(cfg.ny+1);volume=nxy*cfg.nz
    # The source arrays, mutable native reports, Fortran packing/weight work,
    # three VIF height pairs and their first-zero prefix, and graph clocks.
    return {'fddaobs/reports':(24,count),
            'fddaobs/savwt':(6,volume),
            'fddaobs/work':(13 if cfg.obs_scl_neg_qv_innov else 11,volume),
            'fddaobs/planes':(27,nxy),
            'fddaobs/controls':(18+109+(0 if cfg.obs_scl_neg_qv_innov else 2),)}
