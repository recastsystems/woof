"""Distribute native plume scalar-advection loops over the worker warp.

Each native loop completes before its dependent loop. Field copies retain all
200 native elements and the seven prognostic fields retain their native order.
Only the worker warp reaches the warp barriers; the master uses its separate
counted barrier at entry and completion.
"""
import re

NAMES = ('scl_advectc_plumerise', 'fa_zc_plumerise', 'advtndc_plumerise')
FIELDS = ('t', 'qv', 'qc', 'qi', 'qh', 'vel_p', 'rad_p', 'sc')


def functions(text):
    result = {}
    for name in NAMES:
        match = re.search(r'^SDDEV void '+name+r'\([^\n]+\) \{', text, re.M)
        end = text.index('\n}\n', match.start())+3
        result[name] = text[match.start():end]
    return result


def clones(text):
    native = functions(text)
    result = []
    for name, body in native.items():
        if 'arena.alloc' in body or 'state.' in body:
            raise ValueError('Native plume advection acquired shared scratch or state')
        body = body.replace('State &state,Arena &arena,', 'State &state,Arena &arena,int lane,', 1)
        for routine in NAMES:
            body = body.replace(routine+'(', 'sd_worker_'+routine+'(')
            body = body.replace('sd_worker_'+routine+'(state,arena,',
                                'sd_worker_'+routine+'(state,arena,lane,')
        # The only scalar-advection special boundary writes are at level one.
        if name == NAMES[0]:
            if re.findall(r'sd_assign\(coms->scr1,coms->(\w+),arena\);', body) != list(FIELDS):
                raise ValueError('Pinned native scalar copy inventory changed')
            body = body.replace('coms->vt3dc(1) =', 'if(lane==0){\ncoms->vt3dc(1) =', 1)
            body = body.replace('// L1163\n', '// L1163\n}\n__syncwarp(0xffffffffu);\n', 1)
            for field in FIELDS:
                body = body.replace('sd_assign(coms->scr1,coms->'+field+',arena);',
                    'for(int j=lane;j<200;j+=32)coms->scr1.p[j]=coms->'+field+'.p[j];\n__syncwarp(0xffffffffu);')
        body, count = re.subn(r'for \(k=([12]);k<=([^;]+);k\+=1\)',
                             r'for (k=\1+lane;k<=\2;k+=32)', body)
        expected = {NAMES[0]: 2, NAMES[1]: 3, NAMES[2]: 1}[name]
        if count != expected:
            raise ValueError('Pinned native advection loop inventory changed: '+name)
        body, count = re.subn(r'(loop\d+_continue: ; } loop\d+_exit: ; // L\d+)',
                             r'\1\n__syncwarp(0xffffffffu);', body)
        if count != expected:
            raise ValueError('Pinned native advection barrier inventory changed: '+name)
        result.append(body)
    return result


def source(text):
    bodies = clones(text)
    declarations = '\n'.join(body.split(' {\n', 1)[0]+';' for body in bodies)
    prefix = declarations+'\n'+'\n'.join(bodies)+'''
SDDEV void sd_advect_dispatch(Plume *plume,int mzp){
 auto &work=sd_micro_work();work.plume=plume;work.mzp=mzp;work.command=2;
 sd_micro_barrier();sd_micro_barrier();
}
'''
    text = text.replace('volatile int command;', 'volatile int command;int mzp;', 1)
    at = text.index('SDDEV void sd_micro_server(')
    text = text[:at]+prefix+text[at:]
    old = 'sd_micro_worker(state,arena,work,int(threadIdx.x)-32);'
    assert text.count(old) == 1
    text = text.replace(old, '''if(work.command==1)sd_micro_worker(state,arena,work,int(threadIdx.x)-32);
  else sd_worker_scl_advectc_plumerise(state,arena,int(threadIdx.x)-32,work.plume,FStr("SC"),work.mzp);''', 1)
    old = 'scl_advectc_plumerise(state,arena,coms,FStr("SC"),coms->nm1); // L545'
    assert text.count(old) == 1
    return text.replace(old, 'sd_advect_dispatch(coms,coms->nm1); // Native L545', 1)
