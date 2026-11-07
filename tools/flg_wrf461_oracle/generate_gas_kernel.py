"""Closed Fortran statement translator for the WRF FLG gas routines.

Only the listed routines and syntax are accepted. Arithmetic is generated
in source association with WOOF float32 math. Unsupported source fails.
"""
from pathlib import Path
import json
import re
import sys
from extract_tables import statements,split_items

ROUTINES=('gascon','gases','gases_new','qks','qki','qkio3','qopo3s','qoph2o','qopch4','qopn2o','qopo3i','qophc','qopcon',
          'qopo2','qopco2','qopco','qopno','qopch3cl','qopso2','qopcfc11','qopcfc12')
FLOAT_GLOBALS=('umco2','umch4','umn2o','umo2','umno','umso2','umno2','umch3cl','umco','umcfc11','umcfc12')
INT_GLOBALS=('no2s','nco2s','nso2s','nch4s','nnol','nno2l','nso2l','nch3cll','ncos','nn2os','nh2ocs','nh2os','no3s','nh2ol','no3l','nco2l','nn2ol','nch4l','ncfc11l','ncfc12l')
OPS={'.or.':1,'.and.':2,'.eq.':3,'.ne.':3,'.lt.':3,'.le.':3,'.gt.':3,'.ge.':3,'+':4,'-':4,'*':5,'/':5,'**':7}
TOKEN=re.compile(r'\s*(\.(?:or|and|eq|ne|lt|le|gt|ge|true|false)\.|(?:\d+\.\d*|\.\d+|\d+)(?:[ed][+-]?\d+)?|[a-z_]\w*|\*\*|[(),+*/-])',re.I)

class Expression:
    def __init__(self,text,variables,arrays):
        self.tokens=[]; position=0
        while position<len(text):
            match=TOKEN.match(text,position)
            if not match: raise ValueError('unparsed expression '+text[position:])
            self.tokens.append(match[1].lower()); position=match.end()
        self.pos=0; self.variables=variables; self.arrays=arrays
    def take(self):
        token=self.tokens[self.pos]; self.pos+=1; return token
    def parse(self,minbp=0):
        token=self.take()
        if token in ('+','-'):
            value,kind=self.parse(6); left=('('+token+value+')',kind)
        elif token=='(':
            left=self.parse(); assert self.take()==')'
        elif re.match(r'^(\d|\.)',token) and token not in ('.true.','.false.'):
            floating='.' in token or 'e' in token or 'd' in token
            left=(token.replace('d','e')+('f' if floating else ''),'float' if floating else 'int')
        elif token in ('.true.','.false.'):
            left=('true' if token=='.true.' else 'false','bool')
        elif self.pos<len(self.tokens) and self.tokens[self.pos]=='(':
            self.take(); args=[]
            if self.tokens[self.pos]!=')':
                while True:
                    args.append(self.parse())
                    if self.tokens[self.pos]!=',': break
                    self.take()
            assert self.take()==')'
            if token in self.arrays:
                spec=self.arrays[token]
                base,dims=spec[:2]
                array_kind=spec[2] if len(spec)>2 else 'float'
                lowers=spec[3] if len(spec)>3 else ['1']*len(dims)
                if len(args)!=len(dims): raise ValueError('array rank differs: '+token)
                terms=[]; stride='1'
                for (index,kind),extent,lower in zip(args,dims,lowers):
                    if kind!='int': raise ValueError('noninteger array index: '+token)
                    terms.append('('+index+'-'+lower+')*('+stride+')'); stride+='*('+extent+')'
                left=(base+'['+'+'.join(terms)+']',array_kind)
            elif token in ('exp','alog','log','sqrt','abs','int','real'):
                if len(args)!=1: raise ValueError('intrinsic rank differs '+token)
                value,kind=args[0]
                name={'exp':'gfk_exp','alog':'gfk_log','log':'gfk_log','sqrt':'__fsqrt_rn','abs':'fabsf','int':'(int)','real':'(float)'}[token]
                left=(name+'('+value+')','int' if token=='int' else 'float')
            else: raise ValueError('unknown expression call '+token)
        elif token in self.variables: left=self.variables[token]
        elif token in self.arrays: left=(self.arrays[token][0],'array')
        else: raise ValueError('unknown identifier '+token)
        while self.pos<len(self.tokens):
            op=self.tokens[self.pos]
            if op not in OPS or OPS[op]<minbp: break
            self.take(); right=self.parse(OPS[op]+(0 if op=='**' else 1))
            a,ak=left; b,bk=right
            if op in ('.or.','.and.','.eq.','.ne.','.lt.','.le.','.gt.','.ge.'):
                symbol={'.or.':'||','.and.':'&&','.eq.':'==','.ne.':'!=','.lt.':'<','.le.':'<=','.gt.':'>','.ge.':'>='}[op]
                left=('('+a+symbol+b+')','bool')
            elif op=='**':
                if bk=='int' and b=='2': left=('FG_MU('+a+','+a+')','float')
                else: left=('gfk_pow('+a+','+b+')','float')
            elif 'float' in (ak,bk):
                macro={'+':'FG_AD','-':'FG_SU','*':'FG_MU','/':'FG_DV'}[op]
                left=(macro+'('+a+','+b+')','float')
            elif ak==bk=='int': left=('('+a+op+b+')','int')
            else: raise ValueError('invalid arithmetic on '+ak+' '+bk)
        return left
    def complete(self):
        result=self.parse()
        if self.pos!=len(self.tokens): raise ValueError('expression has trailing tokens')
        return result[0]

def extract(source):
    result={}; current=None
    for line in statements(source):
        match=re.match(r'subroutine\s+(\w+)\s*\((.*?)\)',line,re.I)
        if match:
            current=match[1].lower()
            if current in ROUTINES: result[current]={'args':[v.strip().lower() for v in match[2].split(',')],'lines':[]}
        elif re.match(r'end\s*subroutine',line,re.I): current=None
        elif current in ROUTINES: result[current]['lines'].append(line.lower())
    if set(result)!=set(ROUTINES): raise ValueError('the gas routine inventory differs')
    return result

def translate(name,routine,manifest):
    variables={'mbs':('6','int'),'mbir':('12','int'),'mb':('18','int')}
    arrays={}; declarations=[]; types={}; module_aliases={}
    order=list(manifest['tables']); offsets={key:i for i,key in enumerate(order)}
    for i,key in enumerate(FLOAT_GLOBALS): variables[key]=(f'gas_values[{i}]','float')
    for i,key in enumerate(INT_GLOBALS): variables[key]=(f'gas_switches[{i}]','int')
    for line in routine['lines']:
        if line.startswith('use band'):
            module='band_new' if line.startswith('use band_new') else 'band'
            for alias,variable in re.findall(r'(\w+)\s*=>\s*(\w+)',line): module_aliases[alias]=module+'.'+variable
        match=re.match(r'(real|integer)\b(.*)',line)
        if not match: continue
        kind='float' if match[1]=='real' else 'int'; spec=match[2]; attrs=''
        if '::' in spec: attrs,spec=spec.split('::',1)
        common_dims=re.search(r'dimension\(([^)]+)\)',attrs)
        for item in split_items(spec):
            match=re.fullmatch(r'(\w+)\s*(?:\(([^)]+)\))?\s*(?:=\s*(.*))?',item)
            if not match: raise ValueError('unknown declaration '+item)
            var=match[1]; dims=match[2] or (common_dims[1] if common_dims else None); init=match[3]
            types[var]=kind
            if dims:
                shape=[v.strip() for v in dims.split(',')]
                table_key=name+'.'+var
                if table_key in offsets:
                    arrays[var]=(f'(tables+table_offsets[{offsets[table_key]}])',shape)
                else:
                    arrays[var]=(var,shape)
                    if var not in routine['args']:
                        declarations.append(kind+' '+var+'['+'*'.join('101' if d in ('nv','nv1') else d for d in shape)+'];')
            else:
                variables[var]=(var,kind)
                if var not in routine['args']:
                    value='' if init is None else '='+Expression(init,variables,arrays).complete()
                    declarations.append(kind+' '+var+value+';')
    for alias,key in module_aliases.items():
        shape=manifest['tables'][key]['shape']; index=offsets[key]
        if shape: arrays[alias]=(f'(tables+table_offsets[{index}])',[str(v) for v in shape])
        else: variables[alias]=(f'tables[table_offsets[{index}]]','float')
    arguments=[]
    for arg in routine['args']:
        if arg in arrays:
            qualifier='const ' if arg.startswith('coef') or arg in ('pp','ph','po') else ''
            arguments.append(qualifier+'float* '+arg)
        else: arguments.append(('float& ' if arg=='hk' else types[arg]+' ')+arg)
    arguments.extend(['const float* tables','const int* table_offsets','const float* gas_values','const int* gas_switches'])
    signature='__device__ void flg_'+name+'('+','.join(arguments)+')'
    output=declarations[:]; in_switch=False; prior_case=False
    def expression(text): return Expression(text.strip(),variables,arrays).complete()
    for raw in routine['lines']:
        line=re.sub(r'^\d+\s+','',raw).strip()
        if not line or re.match(r'(use |implicit |real\b|integer\b|data\b)',line): continue
        if line=='return': output.append('return;'); continue
        if line=='stop': output.append('return;'); continue
        if line.startswith('select case'):
            output.append('switch('+expression(re.search(r'\((.*)\)',line)[1])+'){'); in_switch=True; continue
        if line.startswith('case'):
            if prior_case: output.append('break;')
            output.append('default:' if line=='case default' else 'case '+re.search(r'\((.*)\)',line)[1]+':'); prior_case=True; continue
        if line=='end select': output.extend(['break;','}']); in_switch=False; continue
        if line in ('endif','end if'): output.append('}'); continue
        if line=='else': output.append('}else{'); continue
        if line.startswith(('elseif','else if')):
            condition=re.search(r'\((.*)\)\s*then',line)[1]; output.append('}else if('+expression(condition)+'){'); continue
        if line.startswith('if'):
            condition=re.search(r'\((.*)\)\s*then',line)
            if not condition: raise ValueError('unsupported inline condition '+line)
            output.append('if('+expression(condition[1])+'){'); continue
        if line.startswith('do while'):
            output.append('while('+expression(re.search(r'\((.*)\)',line)[1])+'){'); continue
        if line.startswith('do '):
            match=re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)',line); limits=list(split_items(match[2]))
            if len(limits)!=2: raise ValueError('unsupported DO step '+line)
            output.append('for('+match[1]+'='+expression(limits[0])+';'+match[1]+'<='+expression(limits[1])+';++'+match[1]+'){'); continue
        if line in ('enddo','end do'): output.append('}'); continue
        if line.startswith('call '):
            match=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',line)
            if match[1] not in ROUTINES: raise ValueError('unsupported call '+line)
            values=[]
            for index,arg in enumerate(split_items(match[2])):
                value=expression(arg)
                if index==2 and match[1] in ('qks','qki','qkio3') and '(' in arg: value='&('+value+')'
                values.append(value)
            values.extend(['tables','table_offsets','gas_values','gas_switches'])
            output.append('flg_'+match[1]+'('+','.join(values)+');'); continue
        if '=' in line:
            left,right=line.split('=',1); output.append(expression(left)+'='+expression(right)+';'); continue
        raise ValueError('unsupported gas statement '+line)
    return signature,signature+'{\n'+'\n'.join(output)+'\n}\n'

def generate(source,manifest):
    routines=extract(source); signatures=[]; bodies=[]
    for name in ROUTINES:
        signature,body=translate(name,routines[name],manifest); signatures.append(signature+';'); bodies.append(body)
    return """// Generated from pristine WRF v4.6.1 FLG gas statements.
#define FG_AD(a,b) __fadd_rn((a),(b))
#define FG_SU(a,b) __fsub_rn((a),(b))
#define FG_MU(a,b) __fmul_rn((a),(b))
#define FG_DV(a,b) __fdiv_rn((a),(b))
"""+ '\n'.join(signatures)+'\n'+'\n'.join(bodies)

if __name__=='__main__':
    source=Path(sys.argv[1]).read_text(); manifest=json.loads(Path(sys.argv[2]).read_text())
    Path(sys.argv[3]).write_text(generate(source,manifest),encoding='utf-8')
