"""Add capture-only statements to a private copy of the pinned WRF source.

The numerical statements of calc_ts and calc_p8w remain unchanged. The binary
capture is a two-by-two cutout of inputs from the first real committed step,
followed by native REAL output words. Use one interior station in one
domain; only its owning MPI rank writes the capture.
"""
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_bytes()
expected = "f8ae2d927e7248efcc9d82dd4de7b85059019edce18f7f9db612da9befebdeb6"
if hashlib.sha256(source).hexdigest() != expected:
    raise SystemExit("refusing a different or already instrumented WRF source")
text = source.decode()
marker = "   INTEGER :: i, k, mm, n, ix, iy, rc\n"
assert text.count(marker) == 1
text = text.replace(marker, marker + "   INTEGER :: capture_unit, cx, cy, ck, cs\n   LOGICAL, SAVE :: captured = .FALSE.\n")
marker = "         grid%ts_tslb(n,i) = grid%tslb(ix,1,iy)"
assert text.count(marker) == 1
capture = r"""
! Capture only. Never change any input or calculated diagnostic.
         IF (.NOT. captured) THEN
           OPEN(NEWUNIT=capture_unit,FILE='tslist-real-input.bin', &
             ACCESS='stream',FORM='unformatted',STATUS='new')
           WRITE(capture_unit) 2, 2, grid%em32-grid%sm32, grid%max_ts_level, &
              grid%use_theta_m, MERGE(1,0,config_flags%tslist_unstagger_winds)
           WRITE(capture_unit) grid%ts_hour(n,i)
           WRITE(capture_unit) (((grid%u_2(cx,ck,cy),cx=ix,ix+2),cy=iy,iy+1),ck=1,grid%em32-1)
           WRITE(capture_unit) (((grid%v_2(cx,ck,cy),cx=ix,ix+1),cy=iy,iy+2),ck=1,grid%em32-1)
           WRITE(capture_unit) (((grid%w_2(cx,ck,cy),cx=ix,ix+1),cy=iy,iy+1),ck=1,grid%em32)
           WRITE(capture_unit) (((grid%phb(cx,ck,cy),cx=ix,ix+1),cy=iy,iy+1),ck=1,grid%em32)
           WRITE(capture_unit) (((grid%ph_2(cx,ck,cy),cx=ix,ix+1),cy=iy,iy+1),ck=1,grid%em32)
           WRITE(capture_unit) (((grid%t_2(cx,ck,cy),cx=ix,ix+1),cy=iy,iy+1),ck=1,grid%em32-1)
           WRITE(capture_unit) (((grid%p(cx,ck,cy)+grid%pb(cx,ck,cy),cx=ix,ix+1),cy=iy,iy+1),ck=1,grid%em32-1)
           WRITE(capture_unit) ((grid%cosa(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%sina(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) grid%fnm(1:grid%em32-1),grid%fnp(1:grid%em32-1)
           WRITE(capture_unit) ((grid%t2(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%q2(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%u10(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%v10(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%psfc(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%glw(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%gsw(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%hfx(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%lh(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%tsk(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%tslb(cx,1,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%rainc(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           WRITE(capture_unit) ((grid%rainnc(cx,cy),cx=ix,ix+1),cy=iy,iy+1)
           DO cs=1,6
             SELECT CASE(cs)
             CASE(1)
               ck=P_QV
             CASE(2)
               ck=P_QC
             CASE(3)
               ck=P_QR
             CASE(4)
               ck=P_QI
             CASE(5)
               ck=P_QS
             CASE(6)
               ck=P_QG
             END SELECT
             IF (ck>=PARAM_FIRST_SCALAR .AND. ck<=num_moist) THEN
               WRITE(capture_unit) (((grid%moist(cx,k,cy,ck),cx=ix,ix+1),cy=iy,iy+1),k=1,grid%em32-1)
             ELSE
               WRITE(capture_unit) (((0.0,cx=ix,ix+1),cy=iy,iy+1),k=1,grid%em32-1)
             END IF
           END DO
           WRITE(capture_unit) grid%ts_hour(n,i),grid%ts_t(n,i),grid%ts_q(n,i), &
             grid%ts_u(n,i),grid%ts_v(n,i),grid%ts_psfc(n,i),grid%ts_glw(n,i),grid%ts_gsw(n,i), &
             grid%ts_hfx(n,i),grid%ts_lh(n,i),grid%ts_tsk(n,i),grid%ts_tslb(n,i), &
             grid%ts_rainc(n,i),grid%ts_rainnc(n,i),grid%ts_clw(n,i)
           WRITE(capture_unit) grid%ts_u_profile(n,i,:),grid%ts_v_profile(n,i,:), &
             grid%ts_w_profile(n,i,:),grid%ts_gph_profile(n,i,:),grid%ts_th_profile(n,i,:), &
             grid%ts_qv_profile(n,i,:),grid%ts_p_profile(n,i,:)
           CLOSE(capture_unit)
           captured=.TRUE.
         END IF
"""
text = text.replace(marker, marker + "\n" + capture)
path.write_text(text)
print("capture instrumentation", path, hashlib.sha256(text.encode()).hexdigest())
