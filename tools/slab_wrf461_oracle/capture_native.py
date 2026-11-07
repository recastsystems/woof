"""Instrument an owned WRF copy without changing SLAB1D arithmetic."""
from pathlib import Path
import hashlib
import sys

path = Path(sys.argv[1]) / "phys/module_sf_slab.F"
source = path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest() == "c44f26777b163234da249373b09a02fa92cb953209bd35a6fe3c30c2158200cb"
declaration = """
    ! Optional proof capture, once per MPI process. No arithmetic is changed.
    logical, save :: slab_capture_done = .false.
    logical :: slab_capture_this
    integer :: slab_capture_in, slab_capture_out, slab_capture_length
    character(len=1024) :: slab_capture_directory, slab_capture_name
"""
anchor = "    REAL    :: THG,ESG,QSG,HFXT,QFXT,CS,CSW,LAMG(4),THCON,PL"
anchor = anchor.strip()
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "\n" + declaration)
start = """
    call get_environment_variable('WOOF_SLAB_CAPTURE_DIR', &
         slab_capture_directory, length=slab_capture_length)
    slab_capture_this = (.not. slab_capture_done .and. slab_capture_length > 0)
    if (slab_capture_this) then
      if (num_soil_layers /= 5) stop 'capture requires the real five-layer slab'
      write(slab_capture_name, '(A,A,I0,A)') &
           trim(slab_capture_directory), '/native-', getpid(), '-input.bin'
      open(newunit=slab_capture_in, file=trim(slab_capture_name), &
           access='stream', form='unformatted', status='new', convert='little_endian')
      write(slab_capture_in) 5150, ite-its+1
      do i=its,ite
        write(slab_capture_in) ifsnow, merge(1,0,radiation), 1, deltsm, &
          t1d(i), qv1d(i), p1d(i), flhc(i), flqc(i), psfcpa(i), &
          xland(i), tmn(i), hfx(i), qfx(i), lh(i), tsk(i), qsfc(i), &
          chklowq(i), gsw(i), glw(i), capg(i), thc(i), snowc(i), &
          emiss(i), mavail(i), tslb2d(i,:), zs, dzs
      end do
      close(slab_capture_in)
      write(slab_capture_name, '(A,A,I0,A)') &
           trim(slab_capture_directory), '/native-', getpid(), '-output.bin'
      open(newunit=slab_capture_out, file=trim(slab_capture_name), &
           access='stream', form='unformatted', status='new', convert='little_endian')
      write(slab_capture_out) 5152, ite-its+1
      write(slab_capture_out) svp1, svp2, svp3, svpt0, ep2, karman, &
                             eomeg, stbolt, xlv, rovcp, p1000mb
    end if
"""
anchor = "    DATA LAMG/1.407E-8, -1.455E-5, 6.290E-3, 0.16857/"
anchor = anchor.strip()
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "\n" + start)
finish = """
    if (slab_capture_this) then
      do i=its,ite
        write(slab_capture_out) hfx(i), qfx(i), lh(i), capg(i), &
          tsk(i), qsfc(i), chklowq(i), tslb2d(i,:), flhc(i), flqc(i)
      end do
      close(slab_capture_out)
      slab_capture_done = .true.
    end if
"""
anchor = "    END SUBROUTINE SLAB1D"
anchor = anchor.strip()
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, finish + "\n" + anchor))
print("instrumented", path, hashlib.sha256(path.read_bytes()).hexdigest())
