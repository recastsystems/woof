"""Capture real first-call BMJDRV rows from a private strict WRF build.

This adds stream writes and local variables, without changing arithmetic.
Only run against an isolated copy. See README.md for the native build and
run settings. Re-running on an already instrumented source is refused.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('wrf_root', type=Path)
parser.add_argument('--receipt', type=Path, required=True)
args = parser.parse_args()
root = args.wrf_root.resolve()
source = root / 'phys/module_cu_bmj.F'
pins = {
    'LICENSE.txt': 'b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6',
    'phys/module_cu_bmj.F': '62d2805aae152b58093d07100000b581458b536164d90f9cecc27a157e2bd92f',
    'share/module_model_constants.F': '5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062',
}
for relative, expected in pins.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'{relative}: {actual}, expected {expected}; source was not changed')
raw = source.read_bytes()
s=raw.decode();decl='''      INTEGER :: CAP_UNIT,CAP_META,CAP_I,CAP_K,CAP_J,CAP_N
      CHARACTER(LEN=160) :: CAP_NAME
      LOGICAL :: CAPTURE_ROW
''';s=s.replace('      INTEGER :: I,J,K,KFLIP,LMH\n','      INTEGER :: I,J,K,KFLIP,LMH\n'+decl,1)
before='''      CAPTURE_ROW = ITIMESTEP.LE.1
      IF(CAPTURE_ROW) THEN
        CAP_J=(JTS+JTE)/2
        CAP_N=ITE-ITS+1
        WRITE(CAP_NAME,'(A,I0,A,I0,A)') 'bmj-native-',ITS,'-',JTS,'.bin'
        OPEN(NEWUNIT=CAP_UNIT,FILE=TRIM(CAP_NAME),ACCESS='STREAM', &
             FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
        OPEN(NEWUNIT=CAP_META,FILE=TRIM(CAP_NAME)//'.txt',STATUS='REPLACE')
        WRITE(CAP_META,*) CAP_N,KTE-KTS+1,DT,STEPCU, &
                          MERGE(1,0,BMJ_RAD_FEEDBACK),ITIMESTEP,ITS,CAP_J
        CLOSE(CAP_META)
'''
for a in ('T','QV','TH','PI','PMID','RHO','DZ8W'):
 before+=f'        WRITE(CAP_UNIT) (({a}(CAP_I,CAP_K,CAP_J),CAP_I=ITS,ITE),CAP_K=KTS,KTE)\n'
before+='        WRITE(CAP_UNIT) ((PINT(CAP_I,CAP_K,CAP_J),CAP_I=ITS,ITE),CAP_K=KTS,KTE+1)\n'
for a in ('XLAND','CLDEFI','KPBL'):
 before+=f'        WRITE(CAP_UNIT) ({a}(CAP_I,CAP_J),CAP_I=ITS,ITE)\n'
before+='''        OPEN(NEWUNIT=CAP_META,FILE=TRIM(CAP_NAME)//'.tables', &
             ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
        WRITE(CAP_META) QS0,SQS,PTBL,THE0,STHE,TTBL,THE0Q,STHEQ,TTBLQ
        CLOSE(CAP_META)
      ENDIF
'''
anchor='      IMD=(IMS+IME)/2';assert s.count(anchor)==1;s=s.replace(anchor,before+anchor)
after='      IF(CAPTURE_ROW) THEN\n'
for a in ('RTHCUTEN','RQVCUTEN','CCLDFRA','QCCONV','QICONV'):
 after+=f'        WRITE(CAP_UNIT) (({a}(CAP_I,CAP_K,CAP_J),CAP_I=ITS,ITE),CAP_K=KTS,KTE)\n'
for a in ('RAINCV','PRATEC','CUTOP','CUBOT','CLDEFI','CONVCLD'):
 after+=f'        WRITE(CAP_UNIT) ({a}(CAP_I,CAP_J),CAP_I=ITS,ITE)\n'
after+='''        WRITE(CAP_UNIT) (MERGE(1,0,CU_ACT_FLAG(CAP_I,CAP_J)),CAP_I=ITS,ITE)
        CLOSE(CAP_UNIT)
      ENDIF
'''
s=s.replace('      END SUBROUTINE BMJDRV',after+'      END SUBROUTINE BMJDRV',1);source.write_text(s)
receipt = {
    'native_root': str(root), 'source_sha256': pins,
    'instrumented_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'capture': 'first call, middle j row of every MPI tile; one tile per rank, OMP_NUM_THREADS=1',
    'layout': 'bmj_oracle_io.read_arm(nstep=1), little endian float32/int32; tables omit expansion scratch',
    'native_flags_required': '-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize',
    'licence': 'WRF public-domain statement in LICENSE.txt; retained in licenses/LICENSE-WRF-public-domain.txt',
}
args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
