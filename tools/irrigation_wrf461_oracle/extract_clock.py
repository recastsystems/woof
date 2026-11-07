"""Keep the two public WRF date routines byte-unmodified in a thin module."""
from pathlib import Path
import hashlib
import re
import sys
source=Path(sys.argv[1]); out=Path(sys.argv[2]); text=source.read_text()
parts=[]
for name in ('get_julgmt','split_date_char'):
    match=re.search(r'^\s*SUBROUTINE '+name+r'\b.*?^\s*END SUBROUTINE '+name+r'\s*$',text,re.M|re.S|re.I)
    if not match:raise ValueError('WRF date routine missing: '+name)
    parts.append(match.group(0))
    print(name,hashlib.sha256(match.group(0).encode()).hexdigest())
out.write_text('MODULE module_date_time\nUSE module_wrf_error\nCONTAINS\n'+'\n'.join(parts)+'\nEND MODULE\n')
