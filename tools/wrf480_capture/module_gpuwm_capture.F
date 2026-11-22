! Read-only argument capture for native WRF module replay (gpuwm tools/wrf480_capture).
! Enabled only when GPUWM_CAPTURE_DIR is non-empty.  GPUWM_CAPTURE_STEPS is a comma
! list of solve_em steps (grid%itimestep after its increment).  Every dump reads its
! argument and writes it to a stream file; nothing is assigned, so the forecast is
! unchanged (proved by comparing the instrumented history with the pristine referee).
! Record: 'GPCR' int32 namelen, name, int32 type (1 r4, 2 r8, 3 i4, 4 logical as i4,
! 5 i8), int32 rank, int32 lbound(rank), int32 shape(rank), data in Fortran order.
MODULE module_gpuwm_capture
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: gpcap_set_step, gpcap_begin, gpcap_end, gpcap_put
  INTEGER, PARAMETER :: MAXN = 64, MAXSTEPS = 256
  LOGICAL, SAVE :: inited = .FALSE., enabled = .FALSE.
  CHARACTER(LEN=1024), SAVE :: capdir = ''
  INTEGER, SAVE :: nsteps = 0, steps(MAXSTEPS), cur_step = -1, unit_ = -1
  INTEGER, SAVE :: nnames = 0, calls(MAXN) = 0, call_step(MAXN) = -1
  CHARACTER(LEN=64), SAVE :: names(MAXN)
  LOGICAL, SAVE :: open_entry(MAXN) = .FALSE.
  INTERFACE gpcap_put
    MODULE PROCEDURE put_r4_0, put_r4_1, put_r4_2, put_r4_3, put_r4_4
    MODULE PROCEDURE put_r8_0, put_r8_1, put_r8_2, put_r8_3, put_r8_4
    MODULE PROCEDURE put_i4_0, put_i4_1, put_i4_2, put_i4_3, put_i4_4
    MODULE PROCEDURE put_i8_0, put_i8_1, put_i8_2
    MODULE PROCEDURE put_l_0, put_l_1, put_l_2, put_l_3
  END INTERFACE
CONTAINS
  SUBROUTINE init_once()
    CHARACTER(LEN=4096) :: s
    INTEGER :: n, i, j, ios, v
    IF (inited) RETURN
    inited = .TRUE.
    CALL get_environment_variable('GPUWM_CAPTURE_DIR', capdir, n)
    enabled = (n > 0)
    IF (.NOT. enabled) RETURN
    CALL get_environment_variable('GPUWM_CAPTURE_STEPS', s, n)
    i = 1
    DO WHILE (i <= n)
      j = INDEX(s(i:n), ',')
      IF (j == 0) THEN
        j = n - i + 2
      END IF
      READ (s(i:i+j-2), *, IOSTAT=ios) v
      IF (ios == 0 .AND. nsteps < MAXSTEPS) THEN
        nsteps = nsteps + 1
        steps(nsteps) = v
      END IF
      i = i + j
    END DO
  END SUBROUTINE init_once

  SUBROUTINE gpcap_set_step(step)
    INTEGER, INTENT(IN) :: step
    CALL init_once()
    cur_step = step
  END SUBROUTINE gpcap_set_step

  INTEGER FUNCTION slot(name)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER :: k
    DO k = 1, nnames
      IF (TRIM(names(k)) == name) THEN
        slot = k
        RETURN
      END IF
    END DO
    nnames = nnames + 1
    names(nnames) = name
    slot = nnames
  END FUNCTION slot

  LOGICAL FUNCTION gpcap_begin(name, phase)
    CHARACTER(LEN=*), INTENT(IN) :: name, phase
    INTEGER :: k, i
    LOGICAL :: hit
    CHARACTER(LEN=1200) :: path
    gpcap_begin = .FALSE.
    CALL init_once()
    IF (.NOT. enabled) RETURN
    hit = .FALSE.
    DO i = 1, nsteps
      IF (steps(i) == cur_step) hit = .TRUE.
    END DO
    IF (.NOT. hit) RETURN
    k = slot(name)
    IF (phase == 'entry') THEN
      IF (call_step(k) /= cur_step) THEN
        call_step(k) = cur_step
        calls(k) = 0
      END IF
      calls(k) = calls(k) + 1
      open_entry(k) = .TRUE.
    ELSE
      IF (.NOT. open_entry(k)) RETURN
      open_entry(k) = .FALSE.
    END IF
    WRITE (path, '(A,"/",A,"-step",I5.5,"-call",I3.3,"-",A,".bin")') &
      TRIM(capdir), name, cur_step, calls(k), phase
    OPEN (NEWUNIT=unit_, FILE=TRIM(path), ACCESS='STREAM', FORM='UNFORMATTED', &
          STATUS='REPLACE', ACTION='WRITE')
    gpcap_begin = .TRUE.
  END FUNCTION gpcap_begin

  SUBROUTINE gpcap_end()
    ! NEWUNIT numbers are negative: test against the -1 sentinel, not > 0
    IF (unit_ /= -1) CLOSE (unit_)
    unit_ = -1
  END SUBROUTINE gpcap_end

  SUBROUTINE hdr(name, t, r, lb, sh)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: t, r, lb(:), sh(:)
    WRITE (unit_) 'GPCR', INT(LEN_TRIM(name), 4), TRIM(name), INT(t, 4), INT(r, 4)
    IF (r > 0) WRITE (unit_) INT(lb(1:r), 4), INT(sh(1:r), 4)
  END SUBROUTINE hdr
  SUBROUTINE put_r4_0(name, a)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(4), INTENT(IN) :: a
    CALL hdr(name, 1, 0, (/ 0 /), (/ 0 /))
    WRITE (unit_) a
  END SUBROUTINE put_r4_0
  SUBROUTINE put_r4_1(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(4), INTENT(IN) :: a(:)
    INTEGER, INTENT(IN) :: lb(1)
    CALL hdr(name, 1, 1, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r4_1
  SUBROUTINE put_r4_2(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(4), INTENT(IN) :: a(:,:)
    INTEGER, INTENT(IN) :: lb(2)
    CALL hdr(name, 1, 2, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r4_2
  SUBROUTINE put_r4_3(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(4), INTENT(IN) :: a(:,:,:)
    INTEGER, INTENT(IN) :: lb(3)
    CALL hdr(name, 1, 3, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r4_3
  SUBROUTINE put_r4_4(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(4), INTENT(IN) :: a(:,:,:,:)
    INTEGER, INTENT(IN) :: lb(4)
    CALL hdr(name, 1, 4, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r4_4
  SUBROUTINE put_r8_0(name, a)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(8), INTENT(IN) :: a
    CALL hdr(name, 2, 0, (/ 0 /), (/ 0 /))
    WRITE (unit_) a
  END SUBROUTINE put_r8_0
  SUBROUTINE put_r8_1(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(8), INTENT(IN) :: a(:)
    INTEGER, INTENT(IN) :: lb(1)
    CALL hdr(name, 2, 1, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r8_1
  SUBROUTINE put_r8_2(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(8), INTENT(IN) :: a(:,:)
    INTEGER, INTENT(IN) :: lb(2)
    CALL hdr(name, 2, 2, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r8_2
  SUBROUTINE put_r8_3(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(8), INTENT(IN) :: a(:,:,:)
    INTEGER, INTENT(IN) :: lb(3)
    CALL hdr(name, 2, 3, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r8_3
  SUBROUTINE put_r8_4(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(8), INTENT(IN) :: a(:,:,:,:)
    INTEGER, INTENT(IN) :: lb(4)
    CALL hdr(name, 2, 4, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_r8_4
  SUBROUTINE put_i4_0(name, a)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(4), INTENT(IN) :: a
    CALL hdr(name, 3, 0, (/ 0 /), (/ 0 /))
    WRITE (unit_) a
  END SUBROUTINE put_i4_0
  SUBROUTINE put_i4_1(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(4), INTENT(IN) :: a(:)
    INTEGER, INTENT(IN) :: lb(1)
    CALL hdr(name, 3, 1, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_i4_1
  SUBROUTINE put_i4_2(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(4), INTENT(IN) :: a(:,:)
    INTEGER, INTENT(IN) :: lb(2)
    CALL hdr(name, 3, 2, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_i4_2
  SUBROUTINE put_i4_3(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(4), INTENT(IN) :: a(:,:,:)
    INTEGER, INTENT(IN) :: lb(3)
    CALL hdr(name, 3, 3, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_i4_3
  SUBROUTINE put_i4_4(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(4), INTENT(IN) :: a(:,:,:,:)
    INTEGER, INTENT(IN) :: lb(4)
    CALL hdr(name, 3, 4, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_i4_4
  SUBROUTINE put_i8_0(name, a)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(8), INTENT(IN) :: a
    CALL hdr(name, 5, 0, (/ 0 /), (/ 0 /))
    WRITE (unit_) a
  END SUBROUTINE put_i8_0
  SUBROUTINE put_i8_1(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(8), INTENT(IN) :: a(:)
    INTEGER, INTENT(IN) :: lb(1)
    CALL hdr(name, 5, 1, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_i8_1
  SUBROUTINE put_i8_2(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER(8), INTENT(IN) :: a(:,:)
    INTEGER, INTENT(IN) :: lb(2)
    CALL hdr(name, 5, 2, lb, SHAPE(a))
    WRITE (unit_) a
  END SUBROUTINE put_i8_2
  SUBROUTINE put_l_0(name, a)
    CHARACTER(LEN=*), INTENT(IN) :: name
    LOGICAL, INTENT(IN) :: a
    CALL hdr(name, 4, 0, (/ 0 /), (/ 0 /))
    WRITE (unit_) MERGE(1_4, 0_4, a)
  END SUBROUTINE put_l_0
  SUBROUTINE put_l_1(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    LOGICAL, INTENT(IN) :: a(:)
    INTEGER, INTENT(IN) :: lb(1)
    CALL hdr(name, 4, 1, lb, SHAPE(a))
    WRITE (unit_) MERGE(1_4, 0_4, a)
  END SUBROUTINE put_l_1
  SUBROUTINE put_l_2(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    LOGICAL, INTENT(IN) :: a(:,:)
    INTEGER, INTENT(IN) :: lb(2)
    CALL hdr(name, 4, 2, lb, SHAPE(a))
    WRITE (unit_) MERGE(1_4, 0_4, a)
  END SUBROUTINE put_l_2
  SUBROUTINE put_l_3(name, a, lb)
    CHARACTER(LEN=*), INTENT(IN) :: name
    LOGICAL, INTENT(IN) :: a(:,:,:)
    INTEGER, INTENT(IN) :: lb(3)
    CALL hdr(name, 4, 3, lb, SHAPE(a))
    WRITE (unit_) MERGE(1_4, 0_4, a)
  END SUBROUTINE put_l_3
END MODULE module_gpuwm_capture
