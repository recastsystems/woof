"""Time both native MM5 surface calls without changing their arithmetic."""
from pathlib import Path
import re,sys
p=Path(sys.argv[1])/'phys/module_surface_driver.F'
s=p.read_text()
for wrapper,callee in (('sfclayrev_seaice_wrapper','sfclayrev'),
                       ('sfclay_seaice_wrapper','sfclay')):
    m=re.search(r'(?ims)^\s*SUBROUTINE '+wrapper+r'\b.*?^\s*END SUBROUTINE '+wrapper,s)
    assert m
    b=m.group(0).replace('     INTEGER :: cap_unit',
        '     REAL :: profile_start,profile_end\n     INTEGER :: profile_unit\n     INTEGER :: cap_unit',1)
    calls=list(re.finditer(r'(?im)^\s*call\s+'+callee+r'\s*\(',b))
    assert len(calls)==2,(wrapper,len(calls))
    for which,call in reversed(list(enumerate(calls))):
        start=call.start();pos=call.end()-1;depth=0
        while pos<len(b):
            end=b.find('\n',pos)
            if end<0:end=len(b)
            line=b[pos:end].split('!',1)[0]
            depth+=line.count('(')-line.count(')')
            if depth==0:break
            pos=end+1
        assert depth==0
        end+=1
        after=f'''     CALL CPU_TIME(profile_end)
     WRITE(cap_file,'(A,I0,A,I0,A,I0,A)') "captures/profile-",ITIMESTEP,"-",ITS,"-",JTS,".txt"
     OPEN(newunit=profile_unit,file=trim(cap_file),status="unknown",position="append")
     WRITE(profile_unit,*) {which+1},profile_end-profile_start
     CLOSE(profile_unit)
'''
        b=b[:start]+'\n     CALL CPU_TIME(profile_start)\n'+b[start:end]+after+b[end:]
    s=s[:m.start()]+b+s[m.end():]
p.write_text(s)
