"""Instrument only an owned pinned PBL driver for the first YSU QNORM call."""
from pathlib import Path
import argparse,hashlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('owned_native_source',type=Path)
source=p.parse_args().owned_native_source
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='90336e30296991fb397ffde87649a4bd20eaa2b7dc6e90639b043810c8420b56'
text=raw.decode();assert text.count('              ENDIF\n!\n! END FASDAS')==1
source.write_text(text.replace('              ENDIF\n!\n! END FASDAS',"                  IF(itimestep==1) THEN\n                    BLOCK\n                    INTEGER :: ncu\n                    OPEN(newunit=ncu,file='qnorm-native.bin',access='stream',form='unformatted',status='new',convert='little_endian')\n                    WRITE(ncu) 'QNORM001', i_end(ij)-i_start(ij)+1, j_end(ij)-j_start(ij)+1, DT\n                    WRITE(ncu) rqvblten(i_start(ij):i_end(ij),1,j_start(ij):j_end(ij))\n                    WRITE(ncu) qv_curr(i_start(ij):i_end(ij),1,j_start(ij):j_end(ij))\n                    WRITE(ncu) QNORM(i_start(ij):i_end(ij),j_start(ij):j_end(ij))\n                    CLOSE(ncu)\n                    END BLOCK\n                  ENDIF\n"+'              ENDIF\n!\n! END FASDAS'))
print(hashlib.sha256(source.read_bytes()).hexdigest())
