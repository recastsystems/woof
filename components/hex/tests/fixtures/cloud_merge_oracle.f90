! WRF v4.6.1 phys/module_radiation_driver.F cloud merge loops.
! Standalone C-binding wrapper; see licenses/LICENSE-WRF-public-domain.txt.
subroutine oracle(n,itimestep,qc,qi,cldfra,qc_bl,qi_bl,cldfra_bl) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n,itimestep
real(c_float)::qc(n,1,1),qi(n,1,1),cldfra(n,1,1),qc_bl(n,1,1),qi_bl(n,1,1),cldfra_bl(n,1,1)
integer::i,j,k,its,ite,jts,jte,kts,kte
its=1
ite=n
jts=1
jte=1
kts=1
kte=1
              IF (itimestep .NE. 1) THEN
                 DO j = jts,jte
                 DO k = kts,kte
                 DO i = its,ite
                    CLDFRA(i,k,j)=CLDFRA_BL(i,k,j)
                 ENDDO
                 ENDDO
                 ENDDO
              ENDIF

              DO j = jts,jte
              DO k = kts,kte
              DO i = its,ite
                 IF (qc(i,k,j) < 1.E-6 .AND. CLDFRA_BL(i,k,j) > 0.001) THEN
                     qc(i,k,j)=qc(i,k,j) + QC_BL(i,k,j)
                 ENDIF
                 IF (qi(i,k,j) < 1.E-8 .AND. CLDFRA_BL(i,k,j) > 0.001) THEN
                    qi(i,k,j)=qi(i,k,j) + QI_BL(i,k,j)
                 ENDIF
              ENDDO
              ENDDO
              ENDDO
end subroutine
