"""Generate the direct column harness from pinned WRF argument declarations."""
from pathlib import Path
import re
import sys


def split(text):
    parts, start, depth = [], 0, 0
    for i, ch in enumerate(text):
        depth += (ch == '(') - (ch == ')')
        if ch == ',' and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    return parts + [text[start:].strip()]


def lines(text):
    pending = ''
    for raw in text.splitlines():
        line = raw.split('!')[0].strip().lower()
        if not line or line.startswith('#'):
            continue
        pending += ' ' + line.lstrip('&').rstrip('&').strip()
        if not line.endswith('&'):
            yield pending.strip()
            pending = ''


def arguments(text, routine):
    body = re.search(r'(?ims)^\s*subroutine ' + routine + r'\s*\(.*?^\s*end subroutine ' + routine, text).group()
    statements = list(lines(body))
    signature = statements[0]
    args = split(signature[signature.index('(') + 1:signature.rindex(')')])
    declarations = {}
    for line in statements[1:]:
        if '::' not in line:
            continue
        attrs, values = line.split('::', 1)
        attrs=re.sub(r'intent\(\s*(\w+)\s*\)',r'intent(\1)',attrs)
        if not re.match(r'(real|integer|logical)\b', attrs):
            continue
        for value in split(values):
            name = re.match(r'\w+', value).group()
            if name not in args:
                continue
            declaration = attrs + ' :: ' + value
            declarations[name] = declaration
    missing = set(args) - set(declarations)
    if missing:
        raise RuntimeError(f'{routine}: undeclared {sorted(missing)}')
    return args, declarations


def runner(source, name, prefix):
    args, declarations = arguments(source, name)
    out = [f'subroutine run_{prefix}(icase)', f'use module_ra_cam, only: {name}' + (', radtpl' if prefix == 'lw' else ''),
           'use module_ra_cam_support, only: cldovrlap', 'implicit none',
           'integer,parameter :: r8=8,pcols=1,ncol=1,pver=32,pverp=33,naer_all=13,nspint=19,naer_groups=7,nbands=2',
           'integer,intent(in) :: icase', 'integer :: k, group', 'real(r8) :: frac, cloud']
    for arg in args:
        if arg in ('pcols', 'ncol', 'pver', 'pverp'):
            continue
        declaration = re.sub(r',?\s*intent\(\w+\)', '', declarations[arg])
        out.append(declaration)
    if prefix=='lw':
        _,paths=arguments(source,'radtpl')
        for key,decl in paths.items():
            if 'intent(out)' in decl:
                out.append(re.sub(r',?\s*intent\(\w+\)', '', decl))
    out += ['group=mod(icase-1,8)', 'cloud=0.0_r8']
    for arg in args:
        if arg in ('pcols', 'ncol', 'pver', 'pverp'):
            continue
        if declarations[arg].startswith('logical'):
            out.append(arg + '=.true.')
        else:
            out.append(arg + '=0')
    out += ['lchnk=1', 'do k=1,pverp', 'frac=real(k-1,r8)/pver',
            'pint(1,k)=exp(log(10000.0_r8)+frac*log((1000000.0_r8-1000.0_r8*(icase-1))/10000.0_r8))',
            'end do', 'do k=1,pver', 'frac=real(k-1,r8)/(pver-1)',
            'pmid(1,k)=0.5_r8*(pint(1,k)+pint(1,k+1))',
            'if(group==1.and.k>pver-10) cloud=1.0_r8',
            'if(group==2.and.((k>=8.and.k<=12).or.(k>=22.and.k<=26))) cloud=0.4_r8+0.1_r8*mod(k,3)',
            'if(group==7.and.k==pver) cloud=1.0e-12_r8', 'cld(1,k)=cloud', 'cloud=0.0_r8']
    if prefix == 'lw':
        out += ['tnm(1,k)=220.0_r8+75.0_r8*frac**2',
                'if(group==5) tnm(1,k)=190.0_r8+65.0_r8*frac**2',
                'if(group==6) tnm(1,k)=230.0_r8+77.0_r8*frac**2',
                'qnm(1,k)=1.0e-6_r8+0.014_r8*frac**5',
                'o3vmr(1,k)=2.0e-8_r8+8.0e-6_r8*exp(-((frac-0.18_r8)/0.15_r8)**2)',
                'n2o(1,k)=0.311e-6_r8*44.0_r8/28.966_r8',
                'ch4(1,k)=1.714e-6_r8*16.0_r8/28.966_r8',
                'cfc11(1,k)=0.280e-9_r8*136.0_r8/28.966_r8',
                'cfc12(1,k)=0.503e-9_r8*120.0_r8/28.966_r8',
                'emis(1,k)=0.98_r8', 'if(group==3) aer_mass(1,k)=0.02_r8', 'end do',
                'pmln=log(pmid)', 'piln=log(pint)',
                'lwupcgs=5.67e-5_r8*(tnm(1,pver)+2.0_r8)**4']
    else:
        out += ['h2ommr(1,k)=1.0e-6_r8+0.014_r8*frac**5',
                'o3mmr(1,k)=(2.0e-8_r8+8.0e-6_r8*exp(-((frac-0.18_r8)/0.15_r8)**2))*48.0_r8/28.966_r8',
                'rh(1,k)=0.2_r8+0.65_r8*frac',
                'cliqwp(1,k)=40.0_r8*cld(1,k)', 'cicewp(1,k)=20.0_r8*cld(1,k)',
                'rel(1,k)=10.0_r8', 'rei(1,k)=30.0_r8',
                'if(group==3) aermmr(1,k,1)=5.0e-7_r8', 'end do',
                'jj=1', 'eccf=1.0_r8', 'scon=1367.0_r8', 'solcon=1367.0',
                'coszrs=0.7_r8', 'if(group==4) coszrs=-0.2_r8',
                'if(group==5) coszrs=0.01_r8', 'if(group==7) coszrs=1.0e-8_r8',
                'asdir=0.15_r8', 'asdif=0.2_r8', 'aldir=0.25_r8', 'aldif=0.3_r8']
    out += ['call cldovrlap(lchnk,ncol,pcols,pver,pverp,pint/10.0_r8,cld,nmxrgn,pmxrgn)']
    out += [f"call dump('{prefix}',icase,'input:nmxrgn',real(nmxrgn,8))"]
    # Inputs are written before the call, so in/out cloud-region arrays cannot contaminate them.
    for arg in args:
        if declarations[arg].startswith('real') and 'intent(out)' not in declarations[arg]:
            out += [f"call dump('{prefix}',icase,'input:{arg}',real(reshape([{arg}],[size([{arg}])]),8))"]
    if prefix=='lw':
        out += ['call radtpl(lchnk,ncol,pcols,pver,pverp,tnm,lwupcgs,qnm,pint,plco2,plh2o, &',
                'tplnka,s2c,tcg,w,tplnke,tint,tint4,tlayr,tlayr4,pmln,piln,plh2ob,wb)']
        for key in ('plco2','plh2o','tplnka','s2c','tcg','w','tplnke','tint','tint4','tlayr','tlayr4','plh2ob','wb'):
            out += [f"call dump('paths',icase,'output:{key}',reshape([{key}],[size({key})]))"]
    out += [f'call {name}(&', ', &\n'.join(','.join(args[i:i+6]) for i in range(0, len(args), 6)) + ')']
    for arg in args:
        if declarations[arg].startswith('real') and 'intent(out)' in declarations[arg]:
            out += [f"call dump('{prefix}',icase,'output:{arg}',real(reshape([{arg}],[size([{arg}])]),8))"]
    out += [f'end subroutine run_{prefix}']
    return '\n'.join(out)


INITIALIZE = '''subroutine init()
use module_ra_cam_support
implicit none
mwdry=28.966
mwco2=44.0
mwh2o=18.016
mwch4=16.0
mwn2o=44.0
mwf11=136.0
mwf12=120.0
allocate(ah2onw(n_p,n_tp,n_u,n_te,n_rh),eh2onw(n_p,n_tp,n_u,n_te,n_rh),ah2ow(n_p,n_tp,n_u,n_te,n_rh))
allocate(cn_ah2ow(n_p,n_tp,n_u,n_te,n_rh),cn_eh2ow(n_p,n_tp,n_u,n_te,n_rh),ln_ah2ow(n_p,n_tp,n_u,n_te,n_rh),ln_eh2ow(n_p,n_tp,n_u,n_te,n_rh))
allocate(ksul(nrh,nspint),wsul(nrh,nspint),gsul(nrh,nspint),ksslt(nrh,nspint),wsslt(nrh,nspint),gsslt(nrh,nspint),kcphil(nrh,nspint),wcphil(nrh,nspint),gcphil(nrh,nspint))
idxSUL=1
idxSSLT=2
idxCARBONfirst=3
idxOCPHO=3
idxBCPHO=4
idxOCPHI=5
idxBCPHI=6
idxDUSTfirst=7
idxBG=11
idxVOLC=12
mxaerl=3
co2mmr=co2vmr*mwco2/mwdry
call radini(9.81,1004.5,0.622,5.67e-8,1013250.0_8)
call dump('constants',1,'input:radtpl',[epsilo,amco2,amd,gravit,p0,rgsslp,co2vmr,stebol,sslp,rga])
call esinti(0.622_8,2.501e6_8,3.336e5_8,461.6_8,1004.5_8,273.16_8)
call aer_optics_initialize
open(33,file='tables.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
open(34,file='tables.tsv',status='replace')
write(34,'(a)') 'field,count,offset_bytes'
call dump_table('ah2onw',reshape([ah2onw],[size(ah2onw)]))
call dump_table('eh2onw',reshape([eh2onw],[size(eh2onw)]))
call dump_table('ah2ow',reshape([ah2ow],[size(ah2ow)]))
call dump_table('cn_ah2ow',reshape([cn_ah2ow],[size(cn_ah2ow)]))
call dump_table('cn_eh2ow',reshape([cn_eh2ow],[size(cn_eh2ow)]))
call dump_table('ln_ah2ow',reshape([ln_ah2ow],[size(ln_ah2ow)]))
call dump_table('ln_eh2ow',reshape([ln_eh2ow],[size(ln_eh2ow)]))
call dump_table('ksul',reshape([ksul],[size(ksul)]))
call dump_table('wsul',reshape([wsul],[size(wsul)]))
call dump_table('gsul',reshape([gsul],[size(gsul)]))
call dump_table('ksslt',reshape([ksslt],[size(ksslt)]))
call dump_table('wsslt',reshape([wsslt],[size(wsslt)]))
call dump_table('gsslt',reshape([gsslt],[size(gsslt)]))
call dump_table('kcphil',reshape([kcphil],[size(kcphil)]))
call dump_table('wcphil',reshape([wcphil],[size(wcphil)]))
call dump_table('gcphil',reshape([gcphil],[size(gcphil)]))
call dump_table('kcphob',reshape([kcphob],[size(kcphob)]))
call dump_table('wcphob',reshape([wcphob],[size(wcphob)]))
call dump_table('gcphob',reshape([gcphob],[size(gcphob)]))
call dump_table('kcb',reshape([kcb],[size(kcb)]))
call dump_table('wcb',reshape([wcb],[size(wcb)]))
call dump_table('gcb',reshape([gcb],[size(gcb)]))
call dump_table('kdst',reshape([kdst],[size(kdst)]))
call dump_table('wdst',reshape([wdst],[size(wdst)]))
call dump_table('gdst',reshape([gdst],[size(gdst)]))
call dump_table('kbg',reshape([kbg],[size(kbg)]))
call dump_table('wbg',reshape([wbg],[size(wbg)]))
call dump_table('gbg',reshape([gbg],[size(gbg)]))
call dump_table('kvolc',reshape([kvolc],[size(kvolc)]))
call dump_table('wvolc',reshape([wvolc],[size(wvolc)]))
call dump_table('gvolc',reshape([gvolc],[size(gvolc)]))
call dump_table('estblh2o',reshape([estblh2o],[size(estblh2o)]))
close(33)
close(34)
end subroutine init'''


def main():
    source = Path(sys.argv[1]).read_text()
    header = '''program cam_columns
implicit none
integer :: icase
open(31,file='columns.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
open(32,file='columns.tsv',status='replace')
write(32,'(a)') 'spectrum,column,field,count,offset_bytes'
call init()
do icase=1,64
call run_lw(icase)
call run_sw(icase)
end do
close(31)
close(32)
contains
subroutine dump_table(field,values)
character(*) :: field
real(8) :: values(:)
integer(8) :: pos
inquire(unit=33,pos=pos)
write(34,'(a,a,i0,a,i0)') field,',',size(values),',',pos-1
write(33) values
end subroutine dump_table
subroutine dump(spectrum,column,field,values)
character(*) :: spectrum,field
integer :: column
real(8) :: values(:)
integer(8) :: pos
inquire(unit=31,pos=pos)
write(32,'(a,a,i0,a,a,a,i0,a,i0)') spectrum,',',column,',',field,',',size(values),',',pos-1
write(31) values
end subroutine dump
'''
    Path(sys.argv[2]).write_text(header + INITIALIZE + '\n' + runner(source, 'radclwmx', 'lw') + '\n' + runner(source, 'radcswmx', 'sw') + '\nend program\n')


if __name__ == '__main__':
    main()
