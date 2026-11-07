"""Reuse native SW transport scratch between sequential spectral intervals.

Each interval completes raddedmx and its flux recurrences before the next
interval begins. Only the selected interval of the ten transport arrays is
live. Keep that plane in lane-interleaved CUDA local memory. Physical table
indices and the native spectral reduction order remain unchanged.
"""
import re
from stage_absorption import function


def compact_shortwave(source):
    transport=('rdir','rdif','tdir','tdif','explay','rdirc','rdifc','tdirc','tdifc','explayc')
    def indices(text):
        for name in transport:
            def index(m):
                value=re.sub(r'\*?ns\b','1',m[1])
                value=re.sub(r'\*\s*19\b','* 1',value)
                value=re.sub(r'-\s*20\b','- 2',value)
                return name+'['+value+']'
            text=re.sub(r'\b'+name+r'\[([^\]]+)\]',index,text)
            text=text.replace(name+'_offset = 1 + 19 *',name+'_offset = 1 + 1 *')
        return text
    first,begin,end=function(source,'radcswmx_')
    body=source[begin:end-1]
    # Native first computes all transport planes, then evaluates their fluxes.
    # Fuse those two independent spectral loops before reusing a plane. The
    # intervening cloud-overlap configuration uses only cld/pmid/pmxrgn and
    # is independent of spectral optics. Its native operations run once before
    # the fused loop. The one-column ABI has ndayc=1 after the night return.
    loop='for (ns = 1; ns <= 19; ++ns) {'
    positions=[m.start() for m in re.finditer(re.escape(loop),body)]
    assert len(positions)==3,positions
    optics_start,second=positions[1:]
    optics_begin=stop=optics_start+len(loop);depth=1
    while depth:depth+=(body[stop]=='{')-(body[stop]=='}');stop+=1
    optics=body[optics_begin:stop-1]
    assert 'raddedmx_(state,' in optics
    # Configurations do not depend on any selected transport plane.
    between=body[stop:second]
    assert not any(re.search(r'\b'+name+r'\[',between) for name in transport)
    body=body[:second+len(loop)]+optics+body[second+len(loop):]
    body=body[:optics_start]+body[stop:]
    for name in transport:
        allocation='*'+name+' = cam_alloc<doublereal>(state, (19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))'
        assert allocation in body,name
        body=body.replace(allocation,name+'[CAM_LEVELS+1] = {}')
    # The flux configuration arrays stay ordered in k/configuration. CUDA
    # local storage interleaves identical indices across column lanes.
    for name in ('udst',):
        match=re.search(r'\*'+name+r' = cam_alloc<(doublereal|integer)>\(state, ',body)
        assert match,name
        pos=stop=match.end();depth=1
        while depth:depth+=(body[stop]=='(')-(body[stop]==')');stop+=1
        body=body[:match.start()]+name+'['+body[pos:stop-1]+'] = {}'+body[stop:]
    body=indices(body)
    source=source[:begin]+body+source[end-1:]
    _,begin,end=function(source,'raddedmx_')
    source=source[:begin]+indices(source[begin:end-1])+source[end-1:]
    return source


def coalesce_flux(source, metadata):
    """Tile the five private flux arrays over column lanes in explicit storage.

    Reuse the original conservative workspace allocation: compaction removes
    18 transport planes and the flux arenas from the ordinary
    call frames. Reserve its tail for the coalesced tile. Static assertions
    emitted below prove that the remaining ordinary frames still fit.
    """
    _,begin,end=function(source,'radcswmx_')
    body=source[begin:end-1];counts=[]
    for name in ('rdndif','rupdif','exptdn','rupdir','tdntot'):
        match=re.search(r'\*'+name+r' = cam_alloc<(doublereal|integer)>\(state, ',body)
        assert match,name
        pos=stop=match.end();depth=1
        while depth:depth+=(body[stop]=='(')-(body[stop]==')');stop+=1
        kind=match[1]
        counts.append(body[pos:stop-1])
        body=body[:match.start()]+'*'+name+' = cam_flux_alloc<'+kind+'>(state, '+body[pos:stop]+body[stop:]
        # Cloud configurations use nested gathers, e.g. cstr[ksort[l-1]].
        # Locate the matching bracket before applying the physical stride.
        for reference in reversed(list(re.finditer(r'\b'+name+r'\[',body))):
            start=stop=reference.end();nested=1
            while nested:
                nested+=(body[stop]=='[')-(body[stop]==']');stop+=1
            stride=('2*' if kind=='integer' else '')+'state->sw_flux_stride'
            body=body[:start]+stride+'*('+body[start:stop-1]+')'+body[stop-1:]
    source=source[:begin]+body+source[end-1:]
    source=source.replace('unsigned long long scratch_offset;',
        'unsigned long long scratch_offset; double* sw_flux; int sw_flux_offset,sw_flux_stride;')
    marker=source.index('__device__ int radems_')
    source=source[:marker]+"""template<typename T> __device__ T* cam_flux_alloc(CamState* s,int n){
        T* p=(T*)(s->sw_flux+s->sw_flux_offset*s->sw_flux_stride);
        s->sw_flux_offset+=n;
        for(int at=0;at<n;++at)p[at*s->sw_flux_stride*(8/sizeof(T))]=0;
        return p;
    }
"""+source[marker:]
    values='('+'+'.join(counts)+')'
    source=source.replace('#define CAM_WORK_BYTES ', '#define CAM_FULL_WORK_BYTES ',1)
    marker=source.index('\n',source.index('#define CAM_FULL_WORK_BYTES '))
    source=source[:marker]+('\n#define CAM_SW_FLUX_VALUES '+values+
        '\n#define CAM_WORK_BYTES (CAM_FULL_WORK_BYTES-CAM_SW_FLUX_VALUES*8)')+source[marker:]
    def setup(m):
        index=m[1];size='workers' if index=='worker' else 'columns'
        return m[0]+f' state->sw_flux_stride=min(32,{size}-({index}/32)*32); state->sw_flux_offset=0; state->sw_flux=(double*)(workspace+{size}*CAM_WORK_BYTES)+({index}/32)*32*CAM_SW_FLUX_VALUES+({index}%32);'
    source,n=re.subn(r'state->scratch=workspace\+(worker|column)\*CAM_WORK_BYTES\+2048;',setup,source)
    assert n>=4,n
    metadata['camrad']['coalesced_flux_values_expression']=values
    return source,metadata


def bound_frames(source,metadata):
    """Conservative simultaneous automatic arrays along every device path."""
    frames={};calls={};arenas={}
    pattern=r'__device__\s+(?:int|integer|logical|doublereal|double)\s+(\w+)\([^;{}]*\)\s*\{'
    for match in re.finditer(pattern,source):
        begin=end=match.end();depth=1
        while depth:depth+=(source[end]=='{')-(source[end]=='}');end+=1
        body=source[begin:end-1]
        counts=re.findall(r'\b\w+\[([^\]]+)\]\s*=\s*\{\}',body)
        frames[match[1]]='+'.join(['0',*(f'(({c})*8)' for c in counts)])
        calls[match[1]]=set(re.findall(r'\b(\w+)\(',body))
        allocations=[]
        for alloc in re.finditer(r'cam_alloc<(\w+)>\(state, ',body):
            pos=stop=alloc.end();nested=1
            while nested:nested+=(body[stop]=='(')-(body[stop]==')');stop+=1
            allocations.append('('+body[pos:stop-1]+')*'+str(8 if alloc[1]=='doublereal' else 4)+'+7')
        arenas[match[1]]='+'.join(['0',*allocations])
    def paths(name,active=()):
        assert name not in active,('recursive CAM device routine',name)
        children=calls[name]&frames.keys()
        tails=[p for child in sorted(children) for p in paths(child,(*active,name))]
        return [frames[name]+'+'+p for p in tails] if tails else [frames[name]]
    metadata['camrad'].pop('local_frame_bound_expression',None)
    metadata['camrad']['local_frame_bound_expressions']=sorted(set(
        expression for name in frames for expression in paths(name)))
    frames=arenas
    bounds=sorted(set(expression for name in frames for expression in paths(name)))
    source+='\n'+'\n'.join('static_assert(CAM_WORK_BYTES>=2048+('+value+'),"CAM main frame exceeds reserved arena");' for value in bounds)+'\n'
    metadata['camrad']['arena_frame_bound_expressions']=bounds
    source=re.sub(r'^#define (CAM_\w*BYTES) (.+)$',
                  lambda m:'#define '+m[1]+' (0ULL+('+m[2]+'))',source,flags=re.M)
    return source,metadata
