"""Run independent native plume microphysics levels on one worker warp.

The original master first prepares levels in native order. This preserves the
mixed old next-level/new previous-level qsat stencil. Microphysics reads only
its own level and cannot change that preparation at a later level.
"""
import re

NAMES = ('waterbal', 'evaporate', 'sublimate', 'glaciate', 'melt', 'convert')
VECTORS = ('cvi', 'est', 'qc', 'qh', 'qi', 'qsat', 'qv', 'rho', 't')
SCALARS = ('dqsdz', 'dt', 'l', 'wbar')
SHARED_BYTES = 2048


def functions(text):
    result = {}
    for name in NAMES:
        match = re.search(r'^SDDEV void ' + name + r'\([^\n]+\) \{', text, re.M)
        end = text.index('\n}\n', match.start()) + 3
        result[name] = text[match.start():end]
    return result


def source(text):
    native = functions(text)
    joined = '\n'.join(native.values())
    vectors = set(re.findall(r'coms->(\w+)\(', joined))
    fields = set(re.findall(r'coms->(\w+)', joined))
    if vectors != set(VECTORS) or fields - vectors != set(SCALARS):
        raise ValueError('Pinned native microphysics field inventory changed')
    # Every vector access must stay on the current level, and no scalar may
    # become a result shared across levels. These are semantic dependencies.
    if any(arg != 'coms->l' for arg in re.findall(r'coms->\w+\(([^()]*)\)', joined)):
        raise ValueError('Native microphysics acquired a neighbor dependency')
    if re.search(r'coms->(?:' + '|'.join(SCALARS) + r')\s*=', joined):
        raise ValueError('Native microphysics acquired a shared scalar result')
    if 'arena.alloc' in joined or 'state.' in joined:
        raise ValueError('Native microphysics acquired shared scratch or module state')
    clones = []
    for body in native.values():
        body = body.replace('Plume* coms', 'SdMicroPlume* coms')
        for name in NAMES:
            body = re.sub(r'\b' + name + r'\(', 'sd_micro_' + name + '(', body)
        clones.append(body)
    prefix = '''
struct SdMicroArray {float *p;SDDEV float &operator()(int k)const{return p[k-1];}};
struct SdMicroPlume {
 int l;float dt,wbar,dqsdz;
''' + ''.join(' SdMicroArray ' + n + ';\n' for n in VECTORS) + '''};
struct SdMicroWork {cuda::barrier<cuda::thread_scope_block> barrier;Plume *plume;volatile int command;float wbar[200],dqsdz[200];};
static_assert(sizeof(SdMicroWork)<=2048,"Plume worker shared allocation");
SDDEV SdMicroWork &sd_micro_work(){
 extern __shared__ __align__(16) unsigned char sd_micro_bytes[];
 return *reinterpret_cast<SdMicroWork*>(sd_micro_bytes);
}
'''
    prefix += r'''
// A standard counted barrier synchronizes the master and 32 worker threads.
// The only block-wide barrier is the uniform initialization handoff.
SDDEV void sd_micro_barrier(){sd_micro_work().barrier.arrive_and_wait();}
'''
    prefix += '\n'.join(body.split(' {\n', 1)[0] + ';' for body in clones) + '\n'
    prefix += '\n'.join(clones)
    prefix += '''
SDDEV void sd_micro_worker(State &state,Arena &arena,SdMicroWork &work,int lane){
 Plume *p=work.plume;
 SdMicroPlume view;
 view.dt=p->dt;
''' + ''.join(' view.' + n + '={p->' + n + '.p};\n' for n in VECTORS) + '''
 for(int k=2+lane;k<=p->nm1-1;k+=32){
  view.l=k;view.wbar=work.wbar[k-1];view.dqsdz=work.dqsdz[k-1];
  sd_micro_waterbal(state,arena,&view);
 }
}
SDDEV void sd_micro_server(State &state,Arena arena){
 auto &work=sd_micro_work();
 for(;;){
  sd_micro_barrier();
  if(work.command==0)return;
  sd_micro_worker(state,arena,work,int(threadIdx.x)-32);
  sd_micro_barrier();
 }
}
SDDEV void sd_micro_dispatch(Plume *plume){
 auto &work=sd_micro_work();work.plume=plume;work.command=1;
 sd_micro_barrier();
 sd_micro_barrier();
}
'''
    at = re.search(r'^SDDEV void makeplume\([^\n]+\) \{', text, re.M).start()
    text = text[:at] + prefix + text[at:]
    old = 'waterbal(state,arena,coms); // L591'
    assert text.count(old) == 1
    text = text.replace(old, '''sd_micro_work().wbar[coms->l-1]=coms->wbar;
sd_micro_work().dqsdz[coms->l-1]=coms->dqsdz; // Native L591 work deferred''', 1)
    old = 'loop3_continue: ; } loop3_exit: ; // L593'
    assert text.count(old) == 1
    text = text.replace(old, old + '\nsd_micro_dispatch(coms);', 1)
    start = text.index('extern "C" SDKERNEL void sd_run_smoke_phase_plume(')
    end = text.index('\n}', start) + 2
    entry = text[start:end]
    entry = entry.replace('if(threadIdx.x)return;', r'''
if(threadIdx.x==0){
 init(&sd_micro_work().barrier,33);
}
__syncthreads();
if(threadIdx.x>0 && threadIdx.x<32)return;
''', 1)
    marker = 'Arena arena{scratch+((long long)lane)*capacity,0,capacity,true,first+lane,first,last};'
    assert entry.count(marker) == 1
    entry = entry.replace(marker, marker + '\nif(threadIdx.x>=32){sd_micro_server(*state,arena);return;}', 1)
    entry = entry[:-2] + '\nsd_micro_work().command=0;sd_micro_barrier();\n}'
    return '#include <cuda/barrier>\n' + text[:start] + entry + text[end:]
