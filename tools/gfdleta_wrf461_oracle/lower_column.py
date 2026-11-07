"""Lower the pinned WRF column statements for a clean CUDA transcription.

The translator is a build tool, not a runtime dependency. This produces a
reviewable Fortran-77 intermediate with fixed per-column storage. Numerical
validation remains against the byte-unmodified Fortran module.
"""
from pathlib import Path
import argparse,re,subprocess,json

SELECT={'ETARA','RADTN','GAUSIN','ZENITH','OZON2D','O3INT','CLO89','LWR88','FST88','E1E290','SPA88','E290','E2SPEC','E3V88','SWR93','RADFS','SOLARD','CAL_MON_DAY'}
MAXL=81

def split_top(s,sep=','):
 out=[];depth=0;start=0;quote=None
 for i,c in enumerate(s):
  if quote:
   if c==quote:quote=None
  elif c in "'\"":quote=c
  elif c=='(':depth+=1
  elif c==')':depth-=1
  elif c==sep and depth==0:out.append(s[start:i].strip());start=i+1
 out.append(s[start:].strip());return out

def logical_lines(text):
 result=[];pending='';label=''
 for raw in text.splitlines():
  if raw.lstrip().startswith('#'):continue
  raw=raw.split('!')[0].strip()
  if not raw:continue
  continued=raw.endswith('&')
  raw=raw.lstrip('&').rstrip('&').strip()
  if not pending:
   m=re.match(r'^(\d+)\s+(.*)',raw)
   if m:label,raw=m.groups()
  pending+=' '+raw
  if not continued:
   for part in split_top(pending.strip(),';'):
    if part:result.append((label,part));label=''
   pending=''
 if pending:raise ValueError('dangling continuation')
 return result

def routines(lines):
 out={};active=None
 for label,s in lines:
  m=re.match(r'^(?:(?:REAL|INTEGER|LOGICAL)\s+)?(?:SUBROUTINE|FUNCTION)\s+(\w+)',s,re.I)
  if m:
   active=m[1].upper();out[active]=[]
  if active:out[active].append((label,s))
  if re.match(r'END\s+(?:SUBROUTINE|FUNCTION)',s,re.I):active=None
 return out

def bounds(dim):
 parts=split_top(dim)
 out=[]
 for p in parts:
  p=p.upper().strip()
  if p in ('IMS:IME','ITS:ITE','JMS:JME','JTS:JTE'):p='1:1'
  elif re.search(r'\bK(?:MS|ME|TS|TE|DS|DE)\b',p):
   a=p.split(':')
   if len(a)==2:
    lo='1' if a[0] in ('KMS','KTS','KDS') else a[0]
    hi=re.sub(r'\b(KME|KTE|KDE)\b',lambda m:'GDMAX+1' if m[0] in ('KME','KDE') else 'GDMAX',a[1])
    p=lo+':'+hi
  elif re.search(r'\b(?:L|LM|LP1|LP2|LP3|LLP1|LL|LM1|LM2|LM3|LLM1|LLM2|LK)\b',p):
   repl={'L':'GDMAX','LM':'GDMAX','LP1':'GDMAX+1','LP2':'GDMAX+2','LP3':'GDMAX+3','LLP1':'2*GDMAX+1','LL':'2*GDMAX','LM1':'GDMAX-1','LM2':'GDMAX-2','LM3':'GDMAX-3','LLM1':'2*GDMAX-1','LLM2':'2*GDMAX-2','LK':'GDMAX'}
   if ':' not in p:p=re.sub(r'\b('+ '|'.join(repl)+r')\b',lambda m:'('+repl[m[0]]+')',p)
  p=re.sub(r'\bKTE\b','GDMAX',p)
  out.append(p)
 return ','.join(out)

def declaration(s):
 if '::' not in s:return None
 lhs,rhs=s.split('::',1);items=split_top(lhs)
 kind=items[0].strip().upper()
 if kind.startswith('REAL (KIND=K15)') or kind.startswith('REAL(KIND=K15)'):kind='DOUBLE PRECISION'
 if kind.startswith('CHARACTER'):kind=kind.replace('CHARACTER(','CHARACTER*').rstrip(')')
 dim=next((v[len('DIMENSION('):-1] for v in items if v.upper().startswith('DIMENSION(')),None)
 param=any(v.upper()=='PARAMETER' for v in items)
 save=any(v.upper()=='SAVE' for v in items)
 decl=[];names=set();extra=[]
 for item in split_top(rhs):
  name=item.split('=')[0].strip();m=re.match(r'([A-Za-z_]\w*)\s*(?:\((.*)\))?$',name)
  if not m:raise ValueError('declaration entity: '+item)
  key=m[1].upper();names.add(key)
  arr=m[2] if m[2] is not None else dim
  name=key+('('+bounds(arr)+')' if arr is not None else '')
  decl.append(name)
  if '=' in item:
   val=item.split('=',1)[1].strip()
   val=val.replace('SELECTED_REAL_KIND(15)','8')
   extra.append(('PARAMETER' if param else 'DATA',key,val))
 result=[kind+' '+','.join(decl)]
 if param:result+=['PARAMETER('+key+'='+val+')' for _,key,val in extra]
 else:result+=['DATA '+key+' /'+val+'/' for _,key,val in extra]
 if save and not param:result+=['SAVE '+','.join(names)]
 return result,names

GLOBALS={
 'LTOP':('INTEGER','3'), 'PTOPC':('REAL','4'), 'XDUO3N':('REAL','37,81'),'XDO3N2':('REAL','37,81'),'XDO3N3':('REAL','37,81'),'XDO3N4':('REAL','37,81'),'PRGFDL':('REAL','81'),
 'AB15WD':('REAL',None),'SKO2D':('REAL',None),'SKC1R':('REAL',None),'SKO3R':('REAL',None),
 'EM1':('REAL','28,180'),'EM1WDE':('REAL','28,180'),'TABLE1':('REAL','28,180'),'TABLE2':('REAL','28,180'),'TABLE3':('REAL','28,180'),'EM3':('REAL','28,180'),
 'SOURCE':('REAL','28,15'),'DSRCE':('REAL','28,15'),
 'T1':('REAL','5040'),'T2':('REAL','5040'),'T4':('REAL','5040'),'EM1V':('REAL','5040'),'EM1VW':('REAL','5040'),'EM3V':('REAL','5040'),
 'AXSD':('REAL','310'),'RSQR':('REAL',None),
 'CO251':('REAL','82,82'),'CDT51':('REAL','82,82'),'C2D51':('REAL','82,82'),'CO258':('REAL','82,82'),'CDT58':('REAL','82,82'),'C2D58':('REAL','82,82'),
 'TBPVS':('REAL','7501'),'C1XPVS':('REAL',None),'C2XPVS':('REAL',None),
}
for n in 'STEMP GTEMP CO231 CO238 C2D31 C2D38 CDT31 CDT38 CO271 CO278 C2D71 C2D78 CDT71 CDT78'.split():GLOBALS[n]=('REAL','82')
for n in 'CO2M51 CO2M58 CDTM51 CDTM58 C2DM51 C2DM58'.split():GLOBALS[n]=('REAL','81')
CLOCK={'R1':('REAL',None),'RSIN1':('REAL',None),'RCOS1':('REAL',None),'RCOS2':('REAL',None)}
PARAM={'NL':('INTEGER','81'),'NBLY':('INTEGER','15'),'RTHRESH':('REAL','1.E-15'),'RTD':('REAL','57.2957763671875'),
 'XSDMAX':('REAL','3.1'),'DXSD':('REAL','.01'),'NXSD':('INTEGER','310'),'SDPRINT':('LOGICAL','.FALSE.'),
 'RHGRD':('REAL','1.'),'T_ICE':('REAL','-40.'),'TRAD_ICE':('REAL','-20.'),'ABSCOEF_W':('REAL','800.'),'ABSCOEF_I':('REAL','500.'),
 'SECANG':('REAL','-1.66'),'CLDCOEF_LW':('REAL','1.5'),'ABSCOEF_LW':('REAL','-2.4900000095367431640625'),
 'QCONV':('REAL','.0001000000047497451305389404296875'),'CTAUCW':('REAL','.080000005662441253662109375'),'CTAUCI':('REAL','.0500000007450580596923828125'),
 'EP_2':('REAL','0.621750414371490478515625'),'RCP':('REAL','0.2857142984867095947265625'),'DEGRAD':('REAL','0.01745329238474369049072265625'),
 'G':('REAL','9.81000041961669921875'),'NX':('INTEGER','7501')}

PARAM.update({'EPSILON': ['REAL', '1.0000000036274937E-15'], 'G': ['REAL', '9.8100004196166992'], 'R_D': ['REAL', '287'], 'CP': ['REAL', '1004.5'], 'R_V': ['REAL', '461.60000610351562'], 'CV': ['REAL', '717.5'], 'CPV': ['REAL', '1846.4000244140625'], 'CVV': ['REAL', '1384.800048828125'], 'CVPM': ['REAL', '-0.71428573131561279'], 'CLIQ': ['REAL', '4190'], 'CICE': ['REAL', '2106'], 'PSAT': ['REAL', '610.780029296875'], 'RCV': ['REAL', '0.40000000596046448'], 'RCP': ['REAL', '0.28571429848670959'], 'ROVG': ['REAL', '29.255859375'], 'C2': ['REAL', '401.80001831054688'], 'MWDRY': ['REAL', '28.965999603271484'], 'P1000MB': ['REAL', '100000'], 'T0': ['REAL', '300'], 'P0': ['REAL', '100000'], 'CPOVCV': ['REAL', '1.3999999761581421'], 'CVOVCP': ['REAL', '0.71428573131561279'], 'RVOVRD': ['REAL', '1.6083624362945557'], 'RERADIUS': ['REAL', '1.569858767425103E-07'], 'ASSELIN': ['REAL', '0.02500000037252903'], 'CB': ['REAL', '25'], 'XLV0': ['REAL', '3150000'], 'XLV1': ['REAL', '2370'], 'XLS0': ['REAL', '2905000'], 'XLS1': ['REAL', '259.53201293945312'], 'XLS': ['REAL', '2850000'], 'XLV': ['REAL', '2500000'], 'XLF': ['REAL', '350000'], 'RHOWATER': ['REAL', '1000'], 'RHOSNOW': ['REAL', '100'], 'RHOAIR0': ['REAL', '1.2799999713897705'], 'RE_QC_BG': ['REAL', '2.4900000425986946E-06'], 'RE_QI_BG': ['REAL', '4.9899999794433825E-06'], 'RE_QS_BG': ['REAL', '9.9899998531327583E-06'], 'RE_QC_MAX': ['REAL', '4.9999998736893758E-05'], 'RE_QI_MAX': ['REAL', '0.00012500000593718141'], 'RE_QS_MAX': ['REAL', '0.00099900003988295794'], 'PICONST': ['REAL', '3.1415927410125732'], 'DEGRAD': ['REAL', '0.01745329238474369'], 'DPD': ['REAL', '0.98630136251449585'], 'SVP1': ['REAL', '0.61119997501373291'], 'SVP2': ['REAL', '17.670000076293945'], 'SVP3': ['REAL', '29.649999618530273'], 'SVPT0': ['REAL', '273.14999389648438'], 'EP_1': ['REAL', '0.60836243629455566'], 'EP_2': ['REAL', '0.62175041437149048'], 'KARMAN': ['REAL', '0.40000000596046448'], 'EOMEG': ['REAL', '7.2921000537462533E-05'], 'STBOLT': ['REAL', '5.6705101059151275E-08'], 'PRANDTL': ['REAL', '0.3333333432674408'], 'W_ALPHA': ['REAL', '0.30000001192092896'], 'W_BETA': ['REAL', '1'], 'PQ0': ['REAL', '379.9051513671875'], 'EPSQ2': ['REAL', '0.20000000298023224'], 'A2': ['REAL', '17.269388198852539'], 'A3': ['REAL', '273.16000366210938'], 'A4': ['REAL', '35.860000610351562'], 'EPSQ': ['REAL', '9.999999960041972E-13'], 'P608': ['REAL', '0.60836243629455566'], 'CLIMIT': ['REAL', '9.9999996826552254E-21'], 'CM1': ['REAL', '2937.39990234375'], 'CM2': ['REAL', '4.9282999038696289'], 'CM3': ['REAL', '23.551799774169922'], 'DEFC': ['REAL', '0'], 'DEFM': ['REAL', '99999'], 'EPSFC': ['REAL', '0.95238101482391357'], 'EPSWET': ['REAL', '0'], 'FCDIF': ['REAL', '0.3333333432674408'], 'FCM': ['REAL', '2.9999999242136255E-05'], 'GMA': ['REAL', '-102.5'], 'P400': ['REAL', '40000'], 'PHITP': ['REAL', '15000'], 'PI2': ['REAL', '6.2831850051879883'], 'PI1': ['REAL', '3.1415925025939941'], 'PLBTM': ['REAL', '105000'], 'PLOMD': ['REAL', '64200'], 'PMDHI': ['REAL', '35000'], 'Q2INI': ['REAL', '0.5'], 'RFCP': ['REAL', '0.0002488800382707268'], 'RHCRIT_LAND': ['REAL', '0.75'], 'RHCRIT_SEA': ['REAL', '0.80000001192092896'], 'RLAG': ['REAL', '14.8125'], 'RLX': ['REAL', '0.89999997615814209'], 'SCQ2': ['REAL', '50'], 'SLOPHT': ['REAL', '0.0010000000474974513'], 'TLC': ['REAL', '1.407944917678833'], 'WA': ['REAL', '0.15000000596046448'], 'WGHT': ['REAL', '0.34999999403953552'], 'WPC': ['REAL', '0.075000002980232239'], 'Z0LAND': ['REAL', '0.10000000149011612'], 'Z0MAX': ['REAL', '0.0080000003799796104'], 'Z0SEA': ['REAL', '0.0010000000474974513'], 'OBLIQUITY': ['REAL', '23.5'], 'ECCENTRICITY': ['REAL', '0.014000000432133675'], 'SEMIMAJORAXIS': ['REAL', '1'], 'ZERO_DATE': ['REAL', '0'], 'EQUINOX_FRACTION': ['REAL', '0']})

PARAM.update({'RTHRESH': ['REAL', '1.0000000036274937E-15'], 'RTD': ['REAL', '57.295780181884766'], 'XSDMAX': ['REAL', '3.0999999046325684'], 'DXSD': ['REAL', '0.0099999997764825821'], 'RHGRD': ['REAL', '1'], 'T_ICE': ['REAL', '-40'], 'TRAD_ICE': ['REAL', '-20'], 'ABSCOEF_W': ['REAL', '800'], 'ABSCOEF_I': ['REAL', '500'], 'SECANG': ['REAL', '-1.6599999666213989'], 'CLDCOEF_LW': ['REAL', '1.5'], 'ABSCOEF_LW': ['REAL', '-2.4900000095367432'], 'QCONV': ['REAL', '9.9999997473787516E-05'], 'CTAUCW': ['REAL', '0.079999998211860657'], 'CTAUCI': ['REAL', '0.049999997019767761']})

def common_specs(names,block):
 out=[];entities=[]
 for n,(kind,dim) in names.items():
  out.append(kind+' '+n+('('+dim+')' if dim else ''));entities.append(n)
 out.append('COMMON /'+block+'/ '+','.join(entities));return out

def emit_fixed(label,s):
 # Fixed format lines. Split only at blanks or punctuation outside literals.
 result=[];prefix=f'{label:>5} '
 while len(s)>66:
  quote=None;cuts=[]
  for i,c in enumerate(s[:66]):
   if quote:
    if c==quote:quote=None
   elif c in "'\"":quote=c
   elif c in ' ,+-*/()':cuts.append(i+1)
  cut=max(cuts) if cuts else 66
  result.append(prefix+s[:cut]);s=s[cut:];prefix='     &'
 result.append(prefix+s);return result

def lower(name,source):
 declared=set();decl=[];body=[];header=source[0][1]
 if '(' in header:declared.update(x.strip().upper() for x in split_top(header.split('(',1)[1].rsplit(')',1)[0]))
 for label,s in source[1:]:
  if s.upper().startswith('IMPLICIT NONE'):continue
  d=declaration(s)
  if d:
   rows,names=d;decl.extend(('',x) for x in rows);declared|=names;continue
  if re.match(r'^(INTEGER|REAL|LOGICAL|DOUBLE PRECISION|DIMENSION|PARAMETER|DATA|EXTERNAL|EQUIVALENCE|SAVE|CHARACTER)\b',s,re.I):

   rawtype=re.match(r'^(INTEGER|REAL|LOGICAL|DOUBLE PRECISION|DIMENSION)\s+(.*)',s,re.I)
   if rawtype:
    for entity in split_top(rawtype[2]):
     key=re.match(r'(\w+)',entity)
     if key:declared.add(key[1].upper())
   if s.upper().startswith('DIMENSION'):
    s='DIMENSION '+','.join(re.sub(r'\((.*)\)',lambda m:'('+bounds(m[1])+')',x) for x in split_top(s[len('DIMENSION'):]))
   decl.append((label,s));continue
  if re.match(r'^(WRITE|PRINT|FORMAT)\b',s,re.I) or re.match(r'^IF\s*\(SDPRINT\)',s,re.I):continue
  s=re.sub(r'PRESENT\(QI\)','.TRUE.',s,flags=re.I)
  if re.match(r'END\s+(SUBROUTINE|FUNCTION)',s,re.I):s='END'
  body.append((label,s))
 text=' '.join(s for _,s in source)
 extra=[('','INTEGER GDMAX'),('','PARAMETER(GDMAX=81)')]
 for n,(kind,value) in PARAM.items():
  if n not in declared and re.search(r'\b'+n+r'\b',text,re.I):extra.extend([('',kind+' '+n),('','PARAMETER('+n+'='+value+')')])
 # Commons have identical declaration order in every unit. A colliding local
 # name is a lowering error, never silently renamed or dropped.
 overlap=declared & (set(GLOBALS)|set(CLOCK))
 if overlap:raise ValueError(name+': common/local collision '+str(overlap))
 extra += [('',x) for x in common_specs(GLOBALS,'GD_TABLES')+common_specs(CLOCK,'GD_CLOCK')]
 out=[]

 nondata=[x for x in decl if not x[1].upper().startswith('DATA ')]
 data=[x for x in decl if x[1].upper().startswith('DATA ')]
 for label,s in [('',header)]+extra+nondata+data+body:out.extend(emit_fixed(label,s))
 return '\n'.join(out)+'\n'

def main():
 p=argparse.ArgumentParser();p.add_argument('wrf',type=Path);p.add_argument('out',type=Path);p.add_argument('f2c',type=Path);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 src=a.wrf/'phys/module_ra_gfdleta.F'
 cpp=subprocess.run(['cpp','-P','-traditional-cpp',str(src)],capture_output=True,text=True,check=True).stdout
 units=routines(logical_lines(cpp))
 mp=routines(logical_lines((a.wrf/'phys/module_mp_etanew.F').read_text()))
 units['FPVS']=mp['FPVS'];select=sorted(SELECT|{'FPVS'})
 audit={}
 for name in select:
  f=a.out/(name.lower()+'.f');f.write_text(lower(name,units[name]));audit[name]={'statements':len(units[name]),'file':f.name}
 (a.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 cmd=[str(a.f2c),'-R','-r','-a','-kr','-A',*[str(a.out/(n.lower()+'.f')) for n in select]]
 r=subprocess.run(cmd,cwd=a.out,capture_output=True,text=True);(a.out/'f2c.log').write_text(r.stdout+r.stderr)
 print(r.stdout+r.stderr);raise SystemExit(r.returncode)

if __name__=='__main__':main()
