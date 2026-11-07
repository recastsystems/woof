! Dump SSiB's module DATA tables exactly as gfortran compiled them, in the
! MODULE_TABLES order of gen_tables.py, as raw little-endian float32.
program dump_tables
  use module_sf_ssib, only : tran0, ref0, green0, vcover0, zlt0, rstpar0, &
       z000, d0, z20, z10, rdc0, rbc0, depth0, soref0, chil0, topt0, tl0, &
       tu0, defac0, ph10, ph20, rootd0, bee0, phsat0, poros0, satco0, slope0
  implicit none
  integer :: u
  open(newunit=u, file='tables-gfortran.bin', access='stream', &
       form='unformatted', status='replace', convert='little_endian')
  write(u) tran0, ref0, green0, vcover0, zlt0, rstpar0, z000, d0, z20, z10, &
       rdc0, rbc0, depth0, soref0, chil0, topt0, tl0, tu0, defac0, ph10, &
       ph20, rootd0, bee0, phsat0, poros0, satco0, slope0
  close(u)
end program dump_tables
