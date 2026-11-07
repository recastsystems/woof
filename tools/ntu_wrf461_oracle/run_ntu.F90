! Drive WRF v4.6.1 module_mp_ntu (mp_physics = 56) over a batch of columns.
!
! Every value this program writes was returned by NTU_INIT or MP_NTU; it only
! builds the WRF-shaped arrays (ims:ime, kms:kme, jms:jme) from the batch file
! make_columns.py wrote and dumps the results.  It never reimplements scheme
! arithmetic.
!
! usage: run_ntu BATCH.bin OUT.bin
program run_ntu
  use module_mp_ntu
  implicit none
  integer, parameter :: nmass = 33
  character(len=4), parameter :: mass_names(nmass) = (/ &
    'TH  ','P   ','DZ  ','PII ','ALT ','QV  ','QC  ','QR  ','QI  ','QS  ', &
    'QG  ','QH  ','NC  ','NR  ','NI  ','NS  ','NG  ','NH  ','QDCN','QTCN', &
    'QCCN','QRCN','QNIN','FI  ','FS  ','VI  ','VS  ','VG  ','AI  ','AS  ', &
    'AG  ','AH  ','I3M ' /)
  integer :: hdr(8), ncol, nz, nsteps, ccnty, irst, istep, f, uin, uout, kme
  logical :: restart
  real :: dt
  real, allocatable :: m(:,:,:,:)        ! (ncol, kme, 1, nmass)
  real, allocatable :: w(:,:,:), phb(:,:,:), ph(:,:,:), pb(:,:,:)
  real, allocatable :: rainnc(:,:), rainncv(:,:), snownc(:,:), snowncv(:,:)
  real, allocatable :: grapnc(:,:), grapncv(:,:), hailnc(:,:), hailncv(:,:)
  real, allocatable :: sr(:,:), xland(:,:), buf2(:,:), buf1(:)
  character(len=1024) :: pin, pout

  call get_command_argument(1, pin)
  call get_command_argument(2, pout)
  if (kind(1.0) /= 4) error stop 3
  open(newunit=uin, file=trim(pin), access='stream', form='unformatted', &
       status='old', action='read')
  read(uin) hdr
  if (hdr(1) /= int(z'4E545531')) error stop 4
  ncol = hdr(2); nz = hdr(3); nsteps = hdr(4); ccnty = hdr(5); irst = hdr(6)
  restart = irst /= 0
  read(uin) dt
  kme = nz + 1
  allocate(m(ncol, kme, 1, nmass), w(ncol, kme, 1), phb(ncol, kme, 1), &
           ph(ncol, kme, 1), pb(ncol, kme, 1))
  allocate(rainnc(ncol,1), rainncv(ncol,1), snownc(ncol,1), snowncv(ncol,1), &
           grapnc(ncol,1), grapncv(ncol,1), hailnc(ncol,1), hailncv(ncol,1), &
           sr(ncol,1), xland(ncol,1), buf2(ncol, kme), buf1(ncol))
  m = 0.; pb = 0.; rainncv = 0.; snowncv = 0.; grapncv = 0.; hailncv = 0.
  sr = 0.
  do f = 1, nmass
    read(uin) buf2(:, 1:nz)
    m(:, 1:nz, 1, f) = buf2(:, 1:nz)
  end do
  read(uin) buf2(:, 1:kme); w(:, :, 1) = buf2
  read(uin) buf2(:, 1:kme); phb(:, :, 1) = buf2
  read(uin) buf2(:, 1:kme); ph(:, :, 1) = buf2
  read(uin) buf1; rainnc(:, 1) = buf1
  read(uin) buf1; snownc(:, 1) = buf1
  read(uin) buf1; grapnc(:, 1) = buf1
  read(uin) buf1; hailnc(:, 1) = buf1
  read(uin) buf1; xland(:, 1) = buf1
  close(uin)

  open(newunit=uout, file=trim(pout), access='stream', form='unformatted', &
       status='replace', action='write')
  ! NTU_INIT: kte = kde (the w-level count), exactly as start_em passes it.
  call ntu_init(phb, ph, m(:,:,:,2), pb, m(:,:,:,5), m(:,:,:,6), &
                m(:,:,:,19), m(:,:,:,20), m(:,:,:,21), m(:,:,:,22), &
                m(:,:,:,23), xland, ccnty, restart, &
                1, ncol + 1, 1, 2, 1, kme, 1, ncol, 1, 1, 1, kme, &
                1, ncol, 1, 1, 1, kme)
  do f = 19, 23
    write(uout) m(:, 1:nz, 1, f)
  end do
  do istep = 1, nsteps
    call mp_ntu(istep, m(:,:,:,1), m(:,:,:,2), m(:,:,:,3), w, m(:,:,:,4), &
                dt, sr, m(:,:,:,6), m(:,:,:,7), m(:,:,:,8), m(:,:,:,9), &
                m(:,:,:,10), m(:,:,:,11), m(:,:,:,12), m(:,:,:,13), &
                m(:,:,:,14), m(:,:,:,15), m(:,:,:,16), m(:,:,:,17), &
                m(:,:,:,18), m(:,:,:,19), m(:,:,:,20), m(:,:,:,21), &
                m(:,:,:,22), m(:,:,:,23), m(:,:,:,24), m(:,:,:,25), &
                m(:,:,:,26), m(:,:,:,27), m(:,:,:,28), m(:,:,:,29), &
                m(:,:,:,30), m(:,:,:,31), m(:,:,:,32), m(:,:,:,33), &
                rainnc, rainncv, snownc, snowncv, grapnc, grapncv, &
                hailnc, hailncv, &
                1, ncol + 1, 1, 2, 1, kme, 1, ncol, 1, 1, 1, kme, &
                1, ncol, 1, 1, 1, nz)
    write(uout) m(:, 1:nz, 1, 1)
    do f = 6, 33
      write(uout) m(:, 1:nz, 1, f)
    end do
    write(uout) rainnc(:,1), rainncv(:,1), snownc(:,1), snowncv(:,1), &
                grapnc(:,1), grapncv(:,1), hailnc(:,1), hailncv(:,1), sr(:,1)
  end do
  close(uout)
end program run_ntu
