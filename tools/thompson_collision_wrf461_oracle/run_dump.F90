program run_dump
  use module_mp_thompson, only: thompson_init, arwen_dump_classic_constants
  implicit none
  real :: hgt(2,2,2)
  hgt = 0.0
  hgt(:,1,:) = 100.0
  hgt(:,2,:) = 200.0
  call thompson_init(hgt=hgt, ids=1, ide=3, jds=1, jde=3, kds=1, kde=2, &
       ims=1, ime=2, jms=1, jme=2, kms=1, kme=2,                        &
       its=1, ite=2, jts=1, jte=2, kts=1, kte=2)
  call arwen_dump_classic_constants()
end program run_dump
