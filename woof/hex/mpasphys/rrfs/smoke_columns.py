"""Column launch lowering for the pinned RRFS smoke graph.

Keep native dummy bounds and strides. Only tile iteration and private automatic
storage are partitioned. This matters for the fork's ims:ims declarations.
"""
import re


def _once(source, old, new):
    if source.count(old)!=1:
        raise ValueError('Pinned SMOKE lowering changed: '+old[:90])
    return source.replace(old,new,1)


def source(text, routines, boundaries, automatic=28):
    text=_once(text,'T *p=nullptr; int rank=0;', 'T *p=nullptr; bool compact=false; int rank=0;')
    text=_once(text,'n*=hi[d]-lo[d]+1;', 'n*=(compact && d==0)?1:hi[d]-lo[d]+1;')
    text=_once(text,'int z=hi[d]-lo[d]+1;', 'int z=(compact && d==0)?1:hi[d]-lo[d]+1;')
    # Plume vectors and private column matrices have one physical dimension.
    # Avoid the general rank loop and integer division at every linear access.
    # Sections with gaps still use the original descriptor walk.
    text=_once(text,'SDDEV T &linear(int n) const {int off=0;',
               'SDDEV T &linear(int n) const {if(rank==1)return p[n*stride[0]];'
               'if(compact && rank==2)return p[n*stride[1]];int off=0;')
    text=_once(text,'r.stride[r.rank++]=stride[d++];', 'if(compact && d==0)r.compact=true;r.stride[r.rank++]=stride[d++];')
    text=_once(text,'int base=stride[0];bool reshape=rank!=r;', 'int base=compact?1:stride[0];bool reshape=compact || rank!=r;')
    text=_once(text,'stride[d]=st;st*=v[2*d+1]-v[2*d]+1;', 'stride[d]=(compact && d==0)?0:st;st*=(compact && d==0)?1:v[2*d+1]-v[2*d]+1;')
    text=_once(text,'int capacity;template<class T,class... A>', 'int capacity;bool column_mode=false;int column=0,ims=0,ime=-1;template<class T,class... A>')
    # Full-grid automatic arrays in the wrapper and diagnostic routine are
    # private to one column. Keep their logical bounds for all nested callees.
    old='auto {name} = arena.alloc<{kind}>(ims,ime,{tail});'
    pattern=r'auto (\w+) = arena.alloc<(float|int)>\(ims,ime,([^;]+)\);'
    matches=list(re.finditer(pattern,text))
    if len(matches)!=automatic:raise ValueError('Pinned full-grid automatic array inventory changed')
    for match in reversed(matches):
        name,kind,tail=match.groups()
        replacement=f'auto {name} = arena.column_mode ? arena.column_alloc<{kind}>(ims,ime,{tail}) : arena.alloc<{kind}>(ims,ime,{tail});'
        text=text[:match.start()]+replacement+text[match.end():]
    marker='used+=words;return r;}};'
    text=_once(text,marker,'''used+=words;return r;}
 template<class T,class... A> SDDEV FArray<T> column_alloc(A... a) {
 FArray<T> r;r.p=(T*)(p+used);r.compact=true;r.bind(a...);
 int words=r.size();if(words<0 || used>capacity-words)sd_capacity_failure();used+=words;return r;}
};''')
    start=text.index('template<class T,class E> SDDEV void sd_assign(')
    end=text.index('\ntemplate<class T> SDDEV T sd_sum',start)
    text=text[:start]+'''template<class E> struct SdScalar {static constexpr bool value=false;};
template<> struct SdScalar<float> {static constexpr bool value=true;};
template<> struct SdScalar<int> {static constexpr bool value=true;};
template<> struct SdScalar<bool> {static constexpr bool value=true;};
template<class T,class E> SDDEV void sd_assign(FArray<T> a,E rhs,Arena &arena){
 ArenaScope scope(arena);auto b=sd_expr(rhs);
 int column_stride=0,period=1;
 if(arena.column_mode && !a.compact)for(int d=0;d<a.rank;d++){
 int n=a.hi[d]-a.lo[d]+1;if(n==arena.ime-arena.ims+1){column_stride=period;break;}period*=n;}
 int count=column_stride?a.size()/(arena.ime-arena.ims+1):a.size();
 if constexpr(SdScalar<E>::value){
 for(int j=0;j<count;j++){int i=column_stride?(j/column_stride)*(column_stride*(arena.ime-arena.ims+1))+(arena.column-arena.ims)*column_stride+j%column_stride:j;a.linear(i)=rhs;}
 return;
 }
 auto t=arena.alloc<T>(1,count);
 for(int j=0;j<count;j++){int i=column_stride?(j/column_stride)*(column_stride*(arena.ime-arena.ims+1))+(arena.column-arena.ims)*column_stride+j%column_stride:j;t.linear(j)=b.linear(i);}
 for(int j=0;j<count;j++){int i=column_stride?(j/column_stride)*(column_stride*(arena.ime-arena.ims+1))+(arena.column-arena.ims)*column_stride+j%column_stride:j;a.linear(i)=t.linear(j);}
}'''+text[end:]
    text=_once(text,'for (i=its;i<=kte;i+=1) { // L914','for (i=its;i<=(arena.column_mode?sd_min(ite,kte):kte);i+=1) { // L914')
    for routine in boundaries.values():
        names=routines[routine]['args']
        bounds = (f'args[{names.index("ims")}].ivalue,last=args[{names.index("ime")}].ivalue'
                  if 'ims' in names else f'args[{names.index("chem_p")}].lo[0],last=args[{names.index("chem_p")}].hi[0]')
        prologue=f'extern "C" SDKERNEL void sd_run_{routine}(State *state,float *scratch,int capacity,Argument *args) {{SDGUARD;Arena arena{{scratch,0,capacity}};'
        setup=f'''extern "C" SDKERNEL void sd_run_{routine}(State *state,float *scratch,int capacity,Argument *args) {{
 int lane=blockIdx.x*blockDim.x+threadIdx.x;
 int first={bounds};
 if(lane>last-first)return;state+=lane;
 Arena arena{{scratch+((long long)lane)*capacity,0,capacity,true,first+lane,first,last}};'''
        text=_once(text,prologue,setup)
        # Restrict loop bounds after loading native scalars, before the graph.
        start=text.index(setup);end=text.index('\n}',start)
        part=text[start:end]
        call=f'\n{routine}(*state,arena,'
        restrict='\nits=sd_max(its,arena.column);ite=sd_min(ite,arena.column);'
        if routine=='smoke_to_mpas_fields':
            restrict='\nits=sd_max(its,arena.column);ite=sd_min(sd_max(ite,kte),arena.column);'
        part=_once(part,call,restrict+call)
        text=text[:start]+part+text[end:]
    text+='''
extern "C" __global__ void sd_initialize_columns(State *state,Plume *plume,float *persistent,int capacity,int columns){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=columns)return;
 Arena arena{persistent+((long long)i)*capacity,0,capacity};initialize(state[i],arena,plume[i]);}
'''
    from .smoke_phases import source as phases
    from .smoke_vectors import source as vectors
    # Native routines pass many rank-four descriptors by value. Inline these
    # boundaries so the compiler can propagate the actual rank and stride.
    text = text.replace('#define SDDEV __device__', '#define SDDEV __device__ __forceinline__', 1)
    from .smoke_micro_levels import source as micro_levels
    from .smoke_advect_levels import source as advect_levels
    return tiles(advect_levels(micro_levels(vectors(phases(text), routines))))


COLUMN_PROLOGUE = ' if(lane>last-first)return;state+=lane;'


def tiles(text):
    """Column tiles: one launch covers columns tile_offset.. of the native tile.

    Every column-mode entry takes the tile's first column as a trailing
    argument. ``lane`` stays the slot in the tile (state, plume and scratch
    storage); the native column is first+lane+tile_offset. Science, loop order
    and arithmetic are unchanged: only which slot holds a column's private
    storage. Breakage it prevents: at 3 km (206,726 columns) one slot per
    column needs ~90 GB of private storage and cannot run beside the model.
    """
    signature = '(State *state,float *scratch,int capacity,Argument *args) {'
    out, start, count = [], 0, 0
    while True:
        at = text.find(COLUMN_PROLOGUE, start)
        if at < 0:
            break
        head = text.rindex('extern "C" SDKERNEL void sd_run_', start, at)
        sig = text.index(signature, head)
        if sig > at:
            raise ValueError('Pinned SMOKE column entry changed')
        arena = 'true,first+lane,first,last};'
        end = text.index(arena, at)
        out.append(text[start:sig])
        out.append('(State *state,float *scratch,int capacity,Argument *args,int tile_offset) {')
        out.append(text[sig + len(signature):at])
        # The first column keeps state slot 0 to itself (later tiles take slots
        # 1..): the wrapper log events exported from slot 0 stay its own.
        out.append(' if(lane+tile_offset>last-first)return;state+=lane+(tile_offset!=0);')
        out.append(text[at + len(COLUMN_PROLOGUE):end])
        out.append('true,first+lane+tile_offset,first,last};')
        start = end + len(arena)
        count += 1
    if count != 9:
        raise ValueError(f'Pinned SMOKE column entry inventory changed: {count}')
    out.append(text[start:])
    return ''.join(out)
