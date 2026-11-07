SUBROUTINE lane_export_aero(path,ccnty)
      CHARACTER(LEN=*), INTENT(IN) :: path
      INTEGER, INTENT(IN) :: ccnty
      INTEGER :: unit
      CALL AERO_CONST(ccnty)
      OPEN(NEWUNIT=unit,FILE=path,ACCESS='STREAM',FORM='UNFORMATTED', &
           STATUS='REPLACE',CONVERT='little_endian')
      WRITE(unit) IBAER,TBLRC,TBLXF,DNC0,DNC1,DNC2,DNC3,SENS,ASH,CMAS,DNAS, &
                  AVAN,BETA1,RXMIN,ZCCN,CNMOD,CNSTD,WMAS,RFACT
      CLOSE(unit)
      END SUBROUTINE lane_export_aero
