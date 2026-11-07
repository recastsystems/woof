"""Normalize pinned CAM numeric routines for a constrained C translation.

The generator refuses array sections or statements it cannot preserve. This is
source tooling, not an alternate numerical implementation of the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from make_driver import lines, split
from normalize_arrays import explicit

KINDS={'real(r8)':'double precision','real':'real','integer':'integer','logical':'logical'}
SHAPES={key:['10','10','25','21','7'] for key in ('ah2onw','eh2onw','ah2ow','cn_ah2ow','cn_eh2ow','ln_ah2ow','ln_eh2ow')}
SHAPES.update({key:['1000','19'] for key in ('ksul','wsul','gsul','ksslt','wsslt','gsslt','kcphil','wcphil','gcphil')})


def declarations(stmt):
    match=re.match(r'^(real\s*\(r8\)|real\b|integer\b|logical\b)\s*(.*)$',stmt)
    if not match:
        return []
    dtype=KINDS[match[1].replace(' ','')]
    tail=match[2]
    attrs=''
    if '::' in tail:
        attrs,tail=tail.split('::',1)
    elif tail.startswith(','):
        return []
    if 'pointer' in attrs:
        return []
    dim=re.search(r'dimension\((.*?)\)',attrs)
    results=[]
    for entity in split(tail):
        base=entity.split('=',1)[0].strip()
        name=re.fullmatch(r'(\w+)\s*(?:\((.*)\))?',base)
        if name:
            shape=split(name[2]) if name[2] else split(dim[1]) if dim else []
            value=entity.split('=',1)[1].strip() if '=' in entity else None
            results.append((name[1],dtype,shape,'parameter' in attrs,value))
    return results


def default_real_literal(match):
    value=match[0]
    if value.lower().endswith('_r8'):
        value=value[:-3]
        if 'e' in value.lower():
            return re.sub('[eE]','d',value)
        return value+'d0'
    if 'd' in value.lower():
        return value
    # F2C's promoted literal must carry the original default-REAL word.
    return format(float(np.float32(value)),'.17e')


def numeric_literals(stmt):
    return re.sub(r'(?<![\w])(?:\d+\.\d*|\.\d+)(?:[ed][+-]?\d+)?(?:_r8)?|(?<![\w])\d+[ed][+-]?\d+(?:_r8)?',default_real_literal,stmt)


def fixed(stmts):
    output=[]
    for stmt in stmts:
        label=re.match(r'^(\d+)\s+',stmt)
        if label:
            prefix=label[1].rjust(5)+' '
            stmt=stmt[label.end():]
        else:
            prefix='      '
        for start in range(0,len(stmt),65):
            output.append((prefix if start==0 else '     &')+stmt[start:start+65])
    return '\n'.join(output)+'\n'


class Translation:
    def __init__(self,source_root,out,f2c):
        self.out=out
        self.f2c=f2c
        self.debug_f90=False
        self.driver=False
        self.sources={}
        self.local_layout={}
        self.params={}
        self.globals={}
        pins={'module_ra_cam':'b0484d38338d9c7925d8011889f5a57280870febe451cfdce843b1e9ae669e32','module_ra_cam_support':'a7f12628b075ed4c2d45e6f3a933149007a5d1a8a0d628111e48b3b7ff394aa5'}
        for module in ('module_ra_cam_support','module_ra_cam'):
            if hashlib.sha256((source_root/'phys'/f'{module}.F').read_bytes()).hexdigest()!=pins[module]:
                raise ValueError('WRF 4.6.1 source pin mismatch: '+module)
            src=subprocess.check_output(['gfortran','-E','-cpp','-ffree-form',str(source_root/'phys'/f'{module}.F')],text=True)
            self.sources[module]=src
            for stmt in lines(re.split(r'(?i)\bcontains\b',src,maxsplit=1)[0]):
                for key,dtype,shape,param,value in declarations(stmt):
                    if param and dtype=='integer':
                        if value and re.fullmatch(r'[0-9 +]+',value):
                            self.params[key]=str(eval(value))
                        continue
                    self.globals[key]=(dtype,shape)
                if stmt.startswith('parameter'):
                    for entry in split(stmt[stmt.index('(')+1:stmt.rindex(')')]):
                        key,value=entry.split('=',1)
                        if re.fullmatch(r'[0-9 +]+',value):
                            self.params[key.strip()]=str(eval(value))
        self.params['r8']='8'
        self.params.update({'ssibscheme':'8','clmscheme':'5'})
        self.names=sorted((k for k in self.globals if k not in self.params),key=lambda k:(self.globals[k][0]!='double precision',self.globals[k][0]!='real',k))
        self.metadata=[]
        offset=0
        for key in self.names:
            dtype,shape=self.globals[key]
            shape=SHAPES.get(key,shape)
            bounds=[]
            for dimension in shape:
                dimension=re.sub(r'\b\w+\b',lambda m:self.params.get(m[0],m[0]),dimension)
                if ':' in dimension:
                    low,high=dimension.split(':')
                else:
                    low,high='1',dimension
                low,high=eval(low),eval(high)
                bounds.append([low,high])
            count=int(np.prod([b-a+1 for a,b in bounds])) if bounds else 1
            size=8 if dtype=='double precision' else 4
            self.metadata.append({'name':key,'dtype':dtype,'bounds':bounds,'count':count,'offset':offset,'bytes':count*size})
            offset+=count*size
        (out/'state-layout.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

    def routine(self,name,owner):
        source=self.sources[owner]
        body=re.search(r'(?ims)^\s*(?:(?:real\(r8\)|logical|integer)\s+)?(?:subroutine|function) '+name+r'\s*\(.*?^\s*end (?:subroutine|function) '+name,source).group()
        statements=list(lines(body))
        if self.driver:
            if name=='trcmix_clwrf':
                stop=next(i for i,s in enumerate(statements) if s.startswith('ptrop =') and 'ncol/2' in s)
                # The remaining tail only formats diagnostics and indexes column zero for ncol=1.
                statements=statements[:stop]+['return','end subroutine '+name]
            statements=[s for s in statements if not s.startswith('call wrf_debug(')]
            statements=['call cam_driver_failure' if re.match(r'call (?:wrf_error_fatal|endrun)\s*\(',s) else s for s in statements]
        signature=statements[0]
        function=bool(re.search(r'\bfunction\b',signature))
        if function:
            signature=('integer function ' if name=='findvalue' else 'logical function ' if name=='validfactors' else 'double precision function ')+signature.split('function ',1)[1]
        args=split(signature[signature.index('(')+1:signature.rindex(')')])
        local={item[0]:item for stmt in statements[1:] for item in declarations(stmt)}
        shapes={k:v[2] for k,v in local.items() if v[2]}
        self.local_layout[name]={k:{'dtype':v[1],'shape':v[2]} for k,v in local.items() if v[2] and k not in args}
        (self.out/'local-layout.json').write_text(json.dumps(self.local_layout,indent=2)+'\n')
        shapes.update({k:v[1] for k,v in self.globals.items() if v[1]})
        statements,loopvars=(statements,set()) if self.debug_f90 else explicit(statements,shapes)
        for key in loopvars:local[key]=(key,'integer',[],False,None)
        shadow=set(local)|set(args)|{name}
        pre=[]
        for key,value in self.params.items():
            if key not in shadow:
                pre.extend(['integer '+key,'parameter ('+key+'='+value+')'])
        for entry in self.metadata:
            shape=[str(high) if low==1 else f'{low}:{high}' for low,high in entry['bounds']]
            pre.append(entry['dtype']+' cg_'+entry['name']+('('+','.join(shape)+')' if shape else ''))
        pre.append('common /cam_state/ '+','.join('cg_'+key for key in self.names))
        decl=['integer '+key for key in sorted(loopvars)]
        code=[]
        for stmt in statements[1:]:
            if stmt.startswith(('use ','character')):continue
            if self.driver and stmt.startswith('print '):continue
            if stmt.startswith('implicit'):
                continue
            if stmt.startswith(('end subroutine','end function')):
                code.append('end')
                continue
            items=declarations(stmt)
            if items:
                for key,dtype,shape,param,value in items:
                    if key==name and function:
                        continue
                    if key not in args:
                        if self.driver:
                            dims={'pverr':'32','pverrp':'33','levsiz':'59','num_months':'13','ppcnst':'3','pcnst':'3','paerlev':'29','naer_c':'13','n_cldadv':'3','ite':'1','its':'1','kte':'32','kts':'1','ims':'1','ime':'1','jms':'1','jme':'1','kms':'1','kme':'33'}
                            shape=[re.sub(r'\b\w+\b',lambda m:dims.get(m[0],m[0]),x) for x in shape]
                        shape=[re.sub(r'\b(?:pverp2|pverp3|pverp4|pverp|pver|pcols)\b',lambda m:{'pcols':'1','pver':'32','pverp':'33','pverp2':'34','pverp3':'35','pverp4':'36'}[m[0]],x) for x in shape]
                    decl.append(dtype+' '+key+('('+','.join(shape)+')' if shape else ''))
                    if param and value:

                        if shape and value.startswith('(/'):
                            decl.append('data '+key+' /'+numeric_literals(value[2:-2])+'/')
                        else:
                            decl.append('parameter ('+key+'='+numeric_literals(value)+')')
                continue
            if stmt.startswith(('parameter','data','save')):
                decl.append(numeric_literals(stmt))
                continue
            if re.search(r'\bwrite\(',stmt):
                if name != 'radcswmx':
                    if self.driver:continue
                    raise RuntimeError('unrecognized CAM diagnostic: '+stmt)
                if stmt.startswith('write('):
                    code.append('call cam_overlap_failure')
                else:
                    code.append(numeric_literals(stmt[:stmt.index('write(')])+'call cam_optics_warning')
                continue
            if ':' in stmt and not self.debug_f90:
                raise RuntimeError(f'{name}: unnormalized array section: {stmt}')
            if self.driver:
                if stmt.startswith('call wrf_debug('):continue
                if stmt.startswith('call wrf_error_fatal('):
                    stmt='call cam_driver_failure'
                stmt=re.sub(r'present\(\w+\)', '.true.',stmt)
                stmt=stmt.replace('wrf_dm_on_monitor()', '.false.')
            stmt=stmt.replace('floor(', 'cam_floor(')
            stmt=re.sub(r',\s*(?:kind\s*=\s*)?r8\s*\)',')',stmt)
            stmt=re.sub(r'\b[a-z]\w*\b',lambda m:'cg_'+m[0] if m[0] in self.globals and m[0] not in shadow and m[0] not in self.params else m[0],stmt)
            code.append(numeric_literals(stmt))
        decl.append('double precision psi,phi,fh2oself,estblf')
        decl.append('integer cam_floor')
        if name!='validfactors':decl.append('logical validfactors')
        decl.sort(key=lambda s:s.startswith(('data','save')))
        decl=[s for s in decl if not(function and s=='double precision psi,phi,fh2oself,estblf')]
        path=self.out/(name+'.f')
        path.write_text(fixed([signature]+pre+decl+code))
        if self.debug_f90:
            path.with_suffix('.f90').write_text('\n'.join([signature]+pre+decl+code)+'\n')
            return
        result=subprocess.run([str(self.f2c),'-R',path.name],cwd=self.out,text=True,capture_output=True)
        (self.out/(name+'.translation.log')).write_text(result.stderr)
        print(name,result.returncode,result.stderr[:1400])
        if result.returncode and not self.debug_f90:
            raise RuntimeError('F2C normalization refused '+name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source_root',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--f2c',type=Path,required=True)
    parser.add_argument('--debug-f90',action='store_true')
    parser.add_argument('--driver',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    translation=Translation(args.source_root,args.out,args.f2c)
    translation.debug_f90=args.debug_f90
    translation.driver=args.driver
    routines=[('radems','module_ra_cam'),('trcplk','module_ra_cam_support'),('trcems','module_ra_cam_support'),('psi','module_ra_cam_support'),('phi','module_ra_cam_support'),('fh2oself','module_ra_cam_support'),('radabs','module_ra_cam'),('trcab','module_ra_cam_support'),('trcabn','module_ra_cam_support'),('radclwmx','module_ra_cam'),('radtpl','module_ra_cam'),('radoz2','module_ra_cam'),('aer_pth','module_ra_cam'),('aer_trn','module_ra_cam'),('sortarray','module_ra_cam_support'),('trcpth','module_ra_cam_support'),('radcswmx','module_ra_cam'),('raddedmx','module_ra_cam'),('findvalue','module_ra_cam_support')]
    if args.driver:
        routines.extend((n,'module_ra_cam') for n in ('param_cldoptics_calc','radctl','camrad','radinp','radozn','oznint','get_aerosol','aerosol_indirect'))
        routines.extend((n,'module_ra_cam_support') for n in ('trcmix_clwrf','aqsat','cldefr','cldems','cldovrlap','getfactors','get_rf_scales','get_int_scales','vert_interpolate','background','scale_aerosols','estblf','validfactors','reltab','reitab','trcmix'))
    for name,owner in routines:
        translation.routine(name,owner)


if __name__=='__main__':
    main()
