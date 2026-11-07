"""Capture a full real second Noah call in an owned strict WRF build."""
from pathlib import Path
import hashlib,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('owned_native_source',type=Path)
p=parser.parse_args().owned_native_source
s=p.read_text()
assert hashlib.sha256(p.read_bytes()).hexdigest()=='bde4ecc9a63c9a57c1ca40408eec1703881749fefd8698a7a1a9260cfaba06b1'
names=('psfc sfcprs sfctmp qv1 qgh dz8w1 glw swdown rainbl sr chs cqs2 chs2 rib vegfra shdmin shdmax tmn xland xice snoalb embck tsk hfx qfx lh grdflx qsfc canwat snow snowc snowh albedo albbck emiss znt z0 snotime lai smstav smstot sfcrunoff udrunoff acsnow acsnom snopcx potevp noahres reslin chklowq').split()
a='its:ite';b='jts:jte'
expr={n:f'{n.upper()}({a},{b})' for n in names}
expr.update(psfc=f'P8W3D({a},1,{b})',sfcprs=f'(P8W3D({a},kts+1,{b})+P8W3D({a},kts,{b}))*0.5',sfctmp=f'T3D({a},1,{b})',qv1=f'QV3D({a},1,{b})',dz8w1=f'DZ8W({a},1,{b})',reslin=f'0.0*TSK({a},{b})')
soil=('smois tslb sh2o smcrel').split()
def dump():
 rows=[f'     WRITE(nfc_unit) {expr[n]}' for n in names]
 rows += [f'     WRITE(nfc_unit) {n.upper()}({a},1:num_soil_layers,{b})' for n in soil]
 return '\n'.join(rows)+'\n'
pos=s.index('   IRRIGATION_CHANNEL =0.0')
decl='''   INTEGER, SAVE :: nfc_count=0
   INTEGER :: nfc_unit
   LOGICAL :: nfc_this
'''
before='''   nfc_count=nfc_count+1
   nfc_this=(nfc_count==2)
   IF(nfc_this) THEN
     IF(its/=ids .OR. jts/=jds .OR. ite/=ide-1 .OR. jte/=jde-1) &
       CALL wrf_error_fatal('Noah frame requires a whole-domain serial native run')
     OPEN(newunit=nfc_unit,file='noah-fasdas-native.bin',access='stream', &
       form='unformatted',status='new',convert='little_endian')
     WRITE(nfc_unit) 'NOAHFAS1'
     WRITE(nfc_unit) ite-its+1,jte-jts+1,itimestep,fasdas,isurban,isice,opt_thcnd
     WRITE(nfc_unit) DT,DZS,XICE_THRESHOLD
     WRITE(nfc_unit) MERGE(1,0,FRPCPN),MERGE(1,0,USEMONALB),MERGE(1,0,RDLAI2D)
'''+dump()+f'''     WRITE(nfc_unit) IVGTYP({a},{b}),ISLTYP({a},{b})
     WRITE(nfc_unit) RHO({a},1,{b}),CPM({a},{b}), &
          SDA_HFX({a},{b}),SDA_QFX({a},{b}),QNORM({a},{b})
   ENDIF
'''
s=s[:pos]+decl+before+s[pos:]
end=s.lower().index('   end subroutine lsm')
s=s[:end]+'   IF(nfc_this) THEN\n'+dump()+'     CLOSE(nfc_unit)\n   ENDIF\n'+s[end:]
p.write_text(s)
print(hashlib.sha256(p.read_bytes()).hexdigest())
