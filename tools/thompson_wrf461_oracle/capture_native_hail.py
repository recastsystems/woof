"""Add output-only wet-column capture to a private strict WRF tree.

The native temperature recorded after the call is the module's t1d, before
the driver's theta/exner round trip. No physics expression is replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PIN = "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"


def instrument(text: str, *, require_graupel: bool = False) -> str:
    def insert(needle: str, replacement: str) -> None:
        nonlocal text
        if text.count(needle) != 1:
            raise ValueError(f"Expected one source anchor: {needle!r}")
        text = text.replace(needle, replacement)

    insert("      CONTAINS\n", """      LOGICAL, PRIVATE, SAVE :: lane_captured = .false.
      LOGICAL, PRIVATE, SAVE :: lane_capture = .false.
      INTEGER, PRIVATE, SAVE :: lane_tend_unit

      CONTAINS
""")
    insert("      integer :: kediagloc\n", """      integer :: kediagloc
      integer :: lane_column_unit, lane_surface_unit
      real :: lane_z
      character(len=128) :: lane_prefix
""")
    needle = "         call mp_thompson(aer_init_opt, wif_input_opt, &"
    insert(needle, """         lane_capture = .false.
         if (is_hail_aware .and. .not. lane_captured .and. itimestep >= 1) then
          if (present(diagflag) .and. present(do_radar_ref)) then
           if (diagflag .and. do_radar_ref == 1) then
            if (maxval(qc1d+qi1d+qr1d+qs1d+qg1d) > 1.e-7) then
             lane_capture = .true.
             lane_captured = .true.
             write(lane_prefix,'(A,I0,A,I0)') 'native-hail-i',i,'-j',j
             open(newunit=lane_column_unit,file=trim(lane_prefix)//'-column.csv', &
                  status='new',action='write')
             write(lane_column_unit,'(A)') &
              'phase,k,z_m,p_pa,pii,w_m_s,dz_m,theta_k,temp_k,qv,qc,qr,qi,qs,qg,'// &
              'ni_per_kg,nr_per_kg,nc_per_kg,nwfa_per_kg,nifa_per_kg,'// &
              'ng_per_kg,qb_m3_per_kg,effc_m,effi_m,effs_m,refl_dbz'
             lane_z = 0.
             do k = kts, kte
              write(lane_column_unit,'(A,I0,24(",",ES24.16E3))') 'before,',k, &
               lane_z+0.5*dz1d(k),p1d(k),pii(i,k,j),w1d(k),dz1d(k), &
               th(i,k,j),t1d(k),qv1d(k),qc1d(k),qr1d(k),qi1d(k),qs1d(k),qg1d(k), &
               ni1d(k),nr1d(k),nc1d(k),nwfa1d(k),nifa1d(k),ng1d(k),qb1d(k), &
               re_cloud(i,k,j),re_ice(i,k,j),re_snow(i,k,j),refl_10cm(i,k,j)
              lane_z = lane_z + dz1d(k)
             enddo
             open(newunit=lane_tend_unit,file=trim(lane_prefix)//'-tendencies.csv', &
                  status='new',action='write')
             write(lane_tend_unit,'(A)') &
              'k,tten,qvten,qcten,qrten,nrten,qiten,niten,qsten,qgten,ngten,'// &
              'qbten,ncten,nwfaten,nifaten'
             open(newunit=lane_surface_unit,file=trim(lane_prefix)//'-metadata.csv', &
                  status='new',action='write')
             write(lane_surface_unit,'(A)') &
              'aer_init_opt,wif_input_opt,itimestep,i,j,has_reqc,has_reqi,has_reqs,diagflag,do_radar_ref'
             write(lane_surface_unit,'(I0,9(",",I0))') &
              aer_init_opt,wif_input_opt,itimestep,i,j,has_reqc,has_reqi,has_reqs, &
              merge(1,0,diagflag),do_radar_ref
             close(lane_surface_unit)
             open(newunit=lane_surface_unit,file=trim(lane_prefix)//'-initial-surface.csv', &
                  status='new',action='write')
             write(lane_surface_unit,'(A)') &
              'rainnc_mm,rainncv_mm,snownc_mm,snowncv_mm,graupelnc_mm,graupelncv_mm,sr'
             write(lane_surface_unit,'(ES24.16E3,6(",",ES24.16E3))') &
              RAINNC(i,j),RAINNCV(i,j),SNOWNC(i,j),SNOWNCV(i,j), &
              GRAUPELNC(i,j),GRAUPELNCV(i,j),SR(i,j)
             close(lane_surface_unit)
            endif
           endif
          endif
         endif

""" + needle)
    needle = "      do k = kts, kte\n         t1d(k)  = t1d(k) + tten(k)*DT"
    insert(needle, """      if (lane_capture) then
       do k = kts, kte
        write(lane_tend_unit,'(I0,14(",",ES24.16E3))') k, &
         tten(k),qvten(k),qcten(k),qrten(k),nrten(k),qiten(k),niten(k), &
         qsten(k),qgten(k),ngten(k),qbten(k),ncten(k),nwfaten(k),nifaten(k)
       enddo
      endif
""" + needle)
    insert("      enddo i_loop\n", """         if (lane_capture) then
          lane_z = 0.
          do k = kts, kte
           write(lane_column_unit,'(A,I0,24(",",ES24.16E3))') 'after,',k, &
            lane_z+0.5*dz1d(k),p1d(k),pii(i,k,j),w1d(k),dz1d(k), &
            th(i,k,j),t1d(k),qv1d(k),qc1d(k),qr1d(k),qi1d(k),qs1d(k),qg1d(k), &
            ni1d(k),nr1d(k),nc1d(k),nwfa1d(k),nifa1d(k),ng1d(k),qb1d(k), &
            re_cloud(i,k,j),re_ice(i,k,j),re_snow(i,k,j),refl_10cm(i,k,j)
           lane_z = lane_z + dz1d(k)
          enddo
          close(lane_column_unit)
          close(lane_tend_unit)
          open(newunit=lane_surface_unit,file=trim(lane_prefix)//'-surface.csv', &
               status='new',action='write')
          write(lane_surface_unit,'(A)') &
           'scenario,dt_s,rainnc_mm,rainncv_mm,snownc_mm,snowncv_mm,'// &
           'graupelnc_mm,graupelncv_mm,sr,nwfa2d_kg_s,nifa2d_kg_s'
          write(lane_surface_unit,'(A,10(",",ES24.16E3))') trim(lane_prefix),dt, &
           RAINNC(i,j),RAINNCV(i,j),SNOWNC(i,j),SNOWNCV(i,j), &
           GRAUPELNC(i,j),GRAUPELNCV(i,j),SR(i,j),nwfa2d(i,j),nifa2d(i,j)
          close(lane_surface_unit)
          lane_capture = .false.
         endif

      enddo i_loop
""")
    if require_graupel:
        text = text.replace(
            "maxval(qc1d+qi1d+qr1d+qs1d+qg1d) > 1.e-7",
            "maxval(qg1d) > 1.e-9 .and. maxval(ng1d) > 0. .and. maxval(qb1d) > 0.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--pristine-source", type=Path,
                        help="Verified original source for reinstrumenting a private tree")
    parser.add_argument("--require-graupel", action="store_true",
                        help="Wait for pre-existing graupel mass, number and volume")
    args = parser.parse_args()
    source = args.wrf_root / "phys/module_mp_thompson.F"
    original = (args.pristine_source or source).read_bytes()
    if hashlib.sha256(original).hexdigest() != PIN:
        raise ValueError("The native Thompson source does not match the public WRF pin")
    output = instrument(original.decode(), require_graupel=args.require_graupel).encode()
    backup = source.with_suffix(".F.lane-pristine")
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("Existing pristine backup differs from the public source")
    else:
        with backup.open("xb") as stream:
            stream.write(original)
    source.write_bytes(output)
    args.receipt.write_text(json.dumps({
        "public_source_sha256": PIN,
        "instrumented_source_sha256": hashlib.sha256(output).hexdigest(),
        "source": str(source), "pristine_backup": str(backup),
        "capture": ("first diagnostic-due column per process with graupel mass, number and volume"
                    if args.require_graupel else
                    "first wet, diagnostic-due column per process, with initial precipitation"),
        "require_graupel": args.require_graupel,
        "temperature_after": "module t1d before driver theta/exner round trip",
        "arithmetic_replacements": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
