"""Every native input loader takes the dycore's defc_a/defc_b decision from the suite row (runtime_v842 helper).

Breakage it prevents: the same-host, launch, LTS and feature loaders passed ``row.get('dycore') is not None``, so the
rrfs-44c0849 row (keep_initialized_deformation false; 44c0849's Registry has no mesh defc_a/defc_b) asked the pool
for words it does not hold and NOAA conus3km hrrrv5 on 8 cards stopped with KeyError ('mesh', 'defc_a').
"""
from pathlib import Path

from woof.hex.mpasphys import runtime_v842 as runtime
from woof.hex.physics_suites import load_table

MPASPHYS = Path(runtime.__file__).resolve().parent


def test_row_flag_decides():
    table = load_table()
    assert runtime.keeps_initialized_deformation(table['mpasphys_rrfs_44c0849']) is False
    assert runtime.keeps_initialized_deformation(table['mpasphys_rrfs']) is True
    assert runtime.keeps_initialized_deformation({}) is False


def test_no_loader_bypasses_the_row_flag():
    for path in MPASPHYS.glob('*.py'):
        assert "keep_initialized_deformation=row.get('dycore') is not None" not in path.read_text(encoding='utf-8'), path.name
