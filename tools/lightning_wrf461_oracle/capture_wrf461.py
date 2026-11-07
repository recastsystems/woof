"""Instrument a private WRF v4.6.1 source copy for one real LPI capture.

Run before building the native executable. This changes only the supplied
source copy, never the CUDA implementation. The capture contains one
16x16 monitor-rank patch around its LPI maximum after one forecast hour.
Only option 3 is supported because its calculation is column local.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json

SOURCE_SHA256 = '97ac20aa241c82d8af11fc9df587c1146fec762d7cd6918d48d3c796adff6980'
DECLARATIONS = ' LOGICAL, SAVE :: lane_captured = .false.\n INTEGER :: lane_u, lane_i, lane_j, lane_i1, lane_j1, lane_nx, lane_ny, lane_nz\n INTEGER :: lane_loc(2)\n LOGICAL, EXTERNAL :: wrf_dm_on_monitor\n'
CAPTURE = " ! Diagnostic-only capture of real native inputs and module output.\n IF (.not.lane_captured .and. curr_secs>=3600.d0 .and. lightning_option==3 .and. wrf_dm_on_monitor()) THEN\n   lane_loc=maxloc(lpi(its:ite,jts:jte))\n   lane_i=max(its,min(ite-15,its+lane_loc(1)-8))\n   lane_j=max(jts,min(jte-15,jts+lane_loc(2)-8))\n   lane_i1=min(ite,lane_i+15);lane_j1=min(jte,lane_j+15)\n   lane_nx=lane_i1-lane_i+1;lane_ny=lane_j1-lane_j+1;lane_nz=kte-kts+1\n   OPEN(newunit=lane_u,file='lane-real-lightning.bin',form='unformatted',access='stream',status='replace',convert='little_endian')\n   WRITE(lane_u) lane_nx,lane_ny,lane_nz,lightning_option,iccg_method,cellcount_method\n   WRITE(lane_u) dt,dx,dy,lightning_dt,flashrate_factor,iccg_prescribed_num,iccg_prescribed_den,cldtop_adjustment\n   WRITE(lane_u) curr_secs\n   WRITE(lane_u) xlat(lane_i:lane_i1,lane_j:lane_j1),xlon(lane_i:lane_i1,lane_j:lane_j1), &\n    xland(lane_i:lane_i1,lane_j:lane_j1),ht(lane_i:lane_i1,lane_j:lane_j1),ktop_deep(lane_i:lane_i1,lane_j:lane_j1)\n   WRITE(lane_u) t_phy(lane_i:lane_i1,kts:kte,lane_j:lane_j1),z(lane_i:lane_i1,kts:kte,lane_j:lane_j1), &\n    w(lane_i:lane_i1,kts:kte,lane_j:lane_j1),refl(lane_i:lane_i1,kts:kte,lane_j:lane_j1), &\n    dz8w(lane_i:lane_i1,kts:kte,lane_j:lane_j1),th_phy(lane_i:lane_i1,kts:kte,lane_j:lane_j1), &\n    pi_phy(lane_i:lane_i1,kts:kte,lane_j:lane_j1)\n   WRITE(lane_u) moist(lane_i:lane_i1,kts:kte,lane_j:lane_j1,P_QV), &\n    moist(lane_i:lane_i1,kts:kte,lane_j:lane_j1,P_QC),moist(lane_i:lane_i1,kts:kte,lane_j:lane_j1,P_QR), &\n    moist(lane_i:lane_i1,kts:kte,lane_j:lane_j1,P_QI),moist(lane_i:lane_i1,kts:kte,lane_j:lane_j1,P_QS), &\n    moist(lane_i:lane_i1,kts:kte,lane_j:lane_j1,P_QG)\n   WRITE(lane_u) iccg_in_num(lane_i:lane_i1,lane_j:lane_j1,:),iccg_in_den(lane_i:lane_i1,lane_j:lane_j1,:)\n   WRITE(lane_u) lpi(lane_i:lane_i1,lane_j:lane_j1),ltngacttime\n   CLOSE(lane_u)\n   lane_captured=.true.\n ENDIF\n"


def instrument(root, patch):
    path = root / 'phys/module_lightning_driver.F'
    original = path.read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f'{path}: expected unchanged public WRF source {SOURCE_SHA256}, got {actual}')
    source = original.decode()
    for needle, addition in (
        (' LOGICAL :: do_ltng\n', DECLARATIONS),
        (" CALL wrf_debug ( 100, ' lightning_driver: returning from')", CAPTURE),
    ):
        if source.count(needle) != 1:
            raise ValueError(f'{path}: native insertion anchor is not unique: {needle!r}')
        source = (source.replace(needle, needle + addition, 1) if needle.endswith('\n')
                  else source.replace(needle, addition + needle, 1))
    patch.write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),
        source.splitlines(True), fromfile='a/phys/module_lightning_driver.F',
        tofile='b/phys/module_lightning_driver.F')))
    path.write_text(source)
    print(json.dumps({'source': str(path), 'original_sha256': actual,
        'instrumented_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'patch': str(patch), 'capture': 'lane-real-lightning.bin'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrf-source', type=Path, required=True)
    parser.add_argument('--patch', type=Path, required=True)
    args = parser.parse_args()
    instrument(args.wrf_source, args.patch)


if __name__ == '__main__':
    main()
