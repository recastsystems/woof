"""Assign independent native collection levels to CUDA threads.

The native gather/scatter chunk is replaced by one selected level. Neighbor
reads are from immutable t7 and w, while every field write uses that level.
The input column context stays unchanged until every level has finished.
"""
from woof.hex.mpasphys.rrfs.nssl_phases import _block


def split_collection(module, entry):
    begin = module.index("__device__ __forceinline__ void f_nssl_2mom_gs(")
    _, end = _block(module, begin)
    level = module[begin:end + 1].replace(
        "void f_nssl_2mom_gs(", "void f_nssl_2mom_level(int selected_level,", 1)
    replacements = {
        "const int ngs = nssl_min(500, nx*nz);": "constexpr int ngs = 1;",
        "kze = ktile;": "kze = selected_level;",
        "((kz=1),0)": "((kz=selected_level),0)",
        "nzmpb = 1;": "nzmpb = selected_level;",
        "numgs = ((((nxz) / (ngs))) + (1));": "numgs = 1;",
    }
    for old, new in replacements.items():
        assert level.count(old) == 1, old
        level = level.replace(old, new)
    module = module[:end + 1] + "\n" + level + module[end + 1:]

    begin = module.index("template<int Phase> __device__ __forceinline__ void f_nssl_2mom_phase(")
    opening, end = _block(module, begin)
    driver = module[begin:end + 1]
    driver = driver.replace(
        "template<int Phase> __device__ __forceinline__ void f_nssl_2mom_phase(",
        "__device__ __forceinline__ void f_nssl_collect_level(int selected_level, NsslArena& collection_arena,")
    opening = driver.index("{")
    driver = driver[:opening + 1] + "\nconstexpr int Phase = 2;\n" + driver[opening + 1:]
    call = driver.index("f_nssl_2mom_gs(")
    call_end = driver.index(";", call) + 1
    invocation = driver[call:call_end].replace("f_nssl_2mom_gs(", "f_nssl_2mom_level(selected_level,")
    driver = driver[:call] + "NsslArena saved_arena = arena; arena = collection_arena;\n" + invocation + "\narena = saved_arena;" + driver[call_end:]
    module = module[:begin] + driver + "\n" + module[begin:]

    begin = entry.index("template<int Phase> __device__ __forceinline__ void nssl_phase_body(")
    _, end = _block(entry, begin)
    body = entry[begin:end + 1]
    body = body.replace(
        "template<int Phase> __device__ __forceinline__ void nssl_phase_body(",
        "__device__ __forceinline__ void nssl_collect_level_body(int selected_level, unsigned char* collection_workspace, unsigned long long collection_stride, int collection_column, NsslPhaseState* output,")
    opening = body.index("{")
    body = body[:opening + 1] + "\nconstexpr int Phase = 2;\n" + body[opening + 1:]
    old = "column.f_nssl_2mom_phase<Phase>(context.driver,substep,"
    new = "NsslDriverState saved_driver=context.driver; NsslArena collection_arena{collection_workspace+collection_stride*(collection_column/32)*32,0,32,collection_column%32,0,collection_stride*3/4}; column.f_nssl_collect_level(selected_level,collection_arena,saved_driver,substep,"
    assert body.count(old) == 1
    body = body.replace(old, new)
    old = "context.globals=column; errors[active]=column.native_error;"
    new = """if(selected_level==1) { output[workspace_column].globals=column; output[workspace_column].driver=saved_driver; }
#ifdef __CUDA_ARCH__
atomicMax(errors+active,column.native_error);
#else
errors[active]=nssl_max(errors[active],column.native_error);
#endif
"""
    assert body.count(old) == 1
    body = body.replace(old, new)
    body = body.replace("if constexpr(Phase != 0) column.native_error=context.globals.native_error;", "column.native_error=context.globals.native_error;")

    begin = entry.index('extern "C" __global__ void nssl_collect(')
    _, end = _block(entry, begin)
    kernel = entry[begin:end + 1]
    kernel = kernel.replace("void nssl_collect(", "void nssl_collect(int point_first,int point_count,unsigned char* collection_workspace,unsigned long long collection_stride,NsslPhaseState* output,")
    old = "int active=first+thread; if(thread<count) nssl_phase_body<2>(substep,contexts,active,thread,"
    new = "int point=point_first+thread; int workspace_column=point%count; int selected_level=point/count+1; int active=first+workspace_column; if(thread<point_count) nssl_collect_level_body(selected_level,collection_workspace,collection_stride,thread,output,substep,contexts,active,workspace_column,"
    assert kernel.count(old) == 1
    kernel = kernel.replace(old, new)
    entry = entry[:begin] + body + "\n" + kernel + entry[end + 1:]
    # Carry errors from every collection level to the subsequent serial phase.
    entry = entry.replace("if constexpr(Phase != 0) column.native_error=context.globals.native_error;",
                          "if constexpr(Phase != 0) column.native_error=nssl_max(context.globals.native_error,errors[active]);")
    # The phased route no longer launches the monolithic kernel. Keep its
    # numerical method in the module for the host witness, but omit unused
    # external entry points from the production image.
    for symbol in ('extern "C" __global__ void nssl_step(',
                   'extern "C" __device__ void nssl_step_body('):
        begin = entry.index(symbol)
        _, end = _block(entry, begin)
        entry = entry[:begin] + entry[end + 1:]
    return module, entry
