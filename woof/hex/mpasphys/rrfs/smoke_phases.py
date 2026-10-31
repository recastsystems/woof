"""Launch the pinned smoke wrapper at its existing science boundaries.

The automatic arrays occupy the same per-column workspace in every phase.
Rebinding their descriptors does not initialize the storage. Native expressions,
loop order and branch predicates stay in the generated graph.
"""

PHASES = ('prepare', 'emissions', 'plume', 'inject', 'deposition', 'diagnostics')


def source(text):
    begin = text.index('SDDEV void mpas_smoke_driver(')
    # Skip the forward declaration.
    begin = text.index('SDDEV void mpas_smoke_driver(', begin + 1)
    end = text.index('\n}\n', begin) + 3
    wrapper = text[begin:end]
    signature, body = wrapper.split(' {\n', 1)
    declarations, body = body.split('errmsg = FStr(""); // L259\n', 1)
    body = 'errmsg = FStr(""); // L259\n' + body[:-2]
    prepare, body = body.split('if (do_mpas_smoke) { // L318\n', 1)
    emissions, body = body.split('if (do_timing) { ; /* Framework timing only. */ } // L450\n', 1)
    plume, body = body.split('if ((state.mpas_smoke_config.addsmoke_flag == 1)) { // L475\n', 1)
    inject, body = body.split('if ((drydep_opt == 1)) { // L552\n', 1)
    deposition, diagnostics = body.split('sd_framework_log(state,595); // L595\n', 1)
    inject = 'if ((state.mpas_smoke_config.addsmoke_flag == 1)) { // L475\n' + inject
    deposition = 'if ((drydep_opt == 1)) { // L552\n' + deposition
    diagnostics = 'sd_framework_log(state,595); // L595\n' + diagnostics
    # The original smoke branch surrounds emissions, plume and injection.
    # Its closing brace in inject stays paired with this new opening brace.
    chunks = (prepare, 'if(do_mpas_smoke){\n' + emissions + '\n}',
              'if(do_mpas_smoke){\n' + plume + '\n}',
              'if(do_mpas_smoke){\n' + inject, deposition, diagnostics)
    common = '''
if(!do_mpas_smoke && !do_mpas_dust && !do_mpas_anthro)return;
if(state.wrapper_failed)return;
errmsg=FStr("");errflg=0;
curr_secs=sd_mul(ktau,dt);julday=int(julian);
do_plumerise=plumerise_opt>0;
call_plume=do_plumerise && plumerisefire_frq>0;
if(call_plume)call_plume=(sd_mod(int(curr_secs),sd_max(1,60*plumerisefire_frq))==0 || ktau==2);
'''
    functions = []
    for name, chunk in zip(PHASES, chunks):
        setup = 'state.wrapper_failed=false;\n' if name == 'prepare' else common
        chunk = chunk.replace('if ((errflg != 0)) { return; }',
                              'if ((errflg != 0)) { state.wrapper_failed=true;return; }')
        functions.append(signature.replace('mpas_smoke_driver(', 'smoke_phase_' + name + '(')
                         + ' {\n' + declarations + setup + chunk + '\n}\n')
    text = text[:end] + '\n'.join(functions) + text[end:]
    text = text.replace('struct State {', 'struct State {\nbool wrapper_failed=false;', 1)
    begin = text.index('extern "C" SDKERNEL void sd_run_mpas_smoke_driver(')
    end = text.index('\n}', begin) + 2
    entry = text[begin:end]
    entries = []
    for name in PHASES:
        part = entry.replace('sd_run_mpas_smoke_driver(', 'sd_run_smoke_phase_' + name + '(')
        part = part.replace('\nmpas_smoke_driver(*state,', '\nsmoke_phase_' + name + '(*state,')
        if name == 'plume':
            # Independent plume solvers get independent blocks. Inactive
            # columns retain the fork's ebu_driver branch and cheap exit.
            part = part.replace('int lane=blockIdx.x*blockDim.x+threadIdx.x;',
                                'if(threadIdx.x)return;int lane=blockIdx.x;')
        entries.append(part)
    return text[:begin] + '\n'.join(entries) + text[end:]
