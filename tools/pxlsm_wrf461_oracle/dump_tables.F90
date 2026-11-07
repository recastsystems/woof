program dump_pxlsm_tables
  ! Write every land-use table of WRF v4.6.1's module_sf_pxlsm_data.F as the
  ! compiled module holds it: the words gfortran put in the object, not a
  ! re-parse of the DATA statements.  gpuwm's packaged
  ! gpuwm/data/pxlsm/pxlsm_tables.json is graded against this dump word for
  ! word (tests/test_pxlsm_tables.py), so a table value cannot drift from
  ! WRF's without a test naming it.
  !
  ! Output: one text line per (dataset, parameter): the dataset name, the
  ! parameter name, the category count, then each value as its IEEE-754
  ! binary32 bit pattern in hexadecimal.
  use module_sf_pxlsm_data
  implicit none
  integer :: u
  open(newunit=u, file='pxlsm-tables.txt', status='replace', action='write')
  call put(u, 'MODIS ', RSMIN_MODIS, Z00_MODIS, VEG0_MODIS, VEGMN0_MODIS, &
           LAI0_MODIS, LAIMN0_MODIS, SNUP0_MODIS, ALBF_MODIS, SNOALB_MODIS)
  call put(u, 'NLCD50', RSMIN_NLCD50, Z00_NLCD50, VEG0_NLCD50, VEGMN0_NLCD50, &
           LAI0_NLCD50, LAIMN0_NLCD50, SNUP0_NLCD50, ALBF_NLCD50, SNOALB_NLCD50)
  call put(u, 'NLCD40', RSMIN_NLCD40, Z00_NLCD40, VEG0_NLCD40, VEGMN0_NLCD40, &
           LAI0_NLCD40, LAIMN0_NLCD40, SNUP0_NLCD40, ALBF_NLCD40, SNOALB_NLCD40)
  call put(u, 'USGS  ', RSMIN_USGS, Z00_USGS, VEG0_USGS, VEGMN0_USGS, &
           LAI0_USGS, LAIMN0_USGS, SNUP0_USGS, ALBF_USGS, SNOALB_USGS)
  call put(u, 'USGS28', RSMIN_USGS28, Z00_USGS28, VEG0_USGS28, VEGMN0_USGS28, &
           LAI0_USGS28, LAIMN0_USGS28, SNUP0_USGS28, ALBF_USGS28, SNOALB_USGS28)
  close(u)
contains
  subroutine put(u, name, rsmin, z00, veg0, vegmn0, lai0, laimn0, snup0, albf, snoalb)
    integer, intent(in) :: u
    character(len=6), intent(in) :: name
    real, intent(in) :: rsmin(:), z00(:), veg0(:), vegmn0(:), lai0(:), laimn0(:)
    real, intent(in) :: snup0(:), albf(:), snoalb(:)
    call row(u, name, 'RSMIN ', rsmin)
    call row(u, name, 'Z00   ', z00)
    call row(u, name, 'VEG0  ', veg0)
    call row(u, name, 'VEGMN0', vegmn0)
    call row(u, name, 'LAI0  ', lai0)
    call row(u, name, 'LAIMN0', laimn0)
    call row(u, name, 'SNUP0 ', snup0)
    call row(u, name, 'ALBF  ', albf)
    call row(u, name, 'SNOALB', snoalb)
  end subroutine put
  subroutine row(u, name, param, values)
    integer, intent(in) :: u
    character(len=6), intent(in) :: name, param
    real, intent(in) :: values(:)
    integer :: k
    write(u, '(A,1X,A,1X,I0)', advance='no') trim(name), trim(param), size(values)
    do k = 1, size(values)
      write(u, '(1X,Z8.8)', advance='no') transfer(values(k), 0)
    end do
    write(u, '(A)') ''
  end subroutine row
end program dump_pxlsm_tables
