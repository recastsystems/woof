"""Table output kinds, skip semantics, precision and device parity."""
from dataclasses import replace
from types import SimpleNamespace
import os
import numpy as np
import pytest
from woof.chem_table import DiagnosticRow
from woof.core.chem_outputs import compile_diagnostic, evaluate
from woof.verify.chem_outputs_reference import evaluate_cpu
from woof.verify.chem_oracle import ulp_table
from chem_pm_support import staged_table, fields_for


@pytest.mark.parametrize('output', ['SMOKE_SFC','SMOKE_COLUMN','DUST_SFC'])
def test_arwen_rows(tmp_path, output):
    table = staged_table(tmp_path)
    fields, alt, dz = fields_for(table)
    diag = next(d for d in table.diagnostics if d.output_name == output)
    out = np.empty(alt.shape[1:], np.float32)
    evaluate(compile_diagnostic(diag, table), SimpleNamespace(**fields), alt, dz, out)
    terms = sum(fields[table.row(n).state_attr].astype(np.float64) for n in diag.terms[0]['species'])
    expected = ((terms / alt.astype(np.float64) * dz.astype(np.float64)).sum(axis=0)*.001
                if diag.kind == 'column_integral' else terms[0] / alt[0].astype(np.float64))
    # At most five species and four levels; 8 float32 eps covers their rounding.
    np.testing.assert_allclose(out, expected, rtol=8*np.finfo(np.float32).eps, atol=0)
    print(output, "max_relative_error", float(np.max(np.abs(out.astype(np.float64)-expected)/np.abs(expected))), "rtol", 8*np.finfo(np.float32).eps)
    assert out.shape == alt.shape[1:]


def test_skipped_terms_and_arbitrary_rows(tmp_path):
    table = staged_table(tmp_path)
    table = replace(table, rows=(table.row('passive_1'),), _by_name=None)
    diag = DiagnosticRow('ANY','ug kg-1','arbitrary row','term_sum_3d',
                         ({'species':('passive_1',),'ops':(('mul',2),('div',4))},
                          {'species':('passive_1','missing'),'ops':()}), 'test','test')
    program = compile_diagnostic(diag, table)
    fields, alt, dz = fields_for(table)
    out = np.empty_like(alt)
    evaluate_cpu(program, fields, alt, dz, out)
    expected = np.divide(np.multiply(fields['chem_passive_1'],np.float32(2)),np.float32(4))
    assert ulp_table(out, expected) == dict(max_ulp=0,n_nonzero=0,n=24)
    assert compile_diagnostic(replace(diag,terms=(diag.terms[1],)),table) is None
    assert len(program.terms) == 1


def test_invalid_output_and_layer_mass(tmp_path):
    table = staged_table(tmp_path)
    fields, alt, dz = fields_for(table)
    diag = next(d for d in table.diagnostics if d.kind == 'column_integral')
    program = compile_diagnostic(diag, table)
    with pytest.raises(ValueError,match='dz8w'):
        evaluate_cpu(program, fields, alt, None, np.empty(alt.shape[1:],np.float32))
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_cpu(program, fields, alt, dz, alt[0])
    with pytest.raises(ValueError,match='out shape'):
        evaluate_cpu(program, fields, alt, dz, np.empty_like(alt))


@pytest.mark.gpu
@pytest.mark.parametrize('output',['PM2_5_DRY','PM10','SMOKE_SFC','SMOKE_COLUMN','DUST_SFC'])
def test_kernel_cpu_bitwise(tmp_path, output):
    if os.environ.get('GPUWM_NO_LOCAL_GPU') == '1':
        pytest.skip('GPUWM_NO_LOCAL_GPU prevents local device access')
    cp = pytest.importorskip('cupy')
    table = staged_table(tmp_path)
    fields, alt, dz = fields_for(table)
    diag = next(d for d in table.diagnostics if d.output_name == output)
    program = compile_diagnostic(diag, table)
    shape = alt.shape if diag.kind == 'term_sum_3d' else alt.shape[1:]
    expected = np.empty(shape,np.float32)
    evaluate_cpu(program,fields,alt,dz,expected)
    out=cp.empty(shape,cp.float32)
    evaluate(program,{k:cp.asarray(v) for k,v in fields.items()},cp.asarray(alt),cp.asarray(dz),out)
    measured=ulp_table(cp.asnumpy(out),expected)
    print(output,measured)
    assert measured == dict(max_ulp=0,n_nonzero=0,n=expected.size)

@pytest.mark.parametrize('kind',['term_sum_3d','term_sum_surface','column_integral'])
@pytest.mark.parametrize('divide',[False,True])
def test_all_kinds_scalar_word_reference(tmp_path, kind, divide):
    table=staged_table(tmp_path)
    a,b=table.rows[:2]
    diag=DiagnosticRow('ANY','test','ordered ops',kind,
                       ({'species':(a.name,b.name),'ops':(('mul',.38),('div',1.375))},
                        {'species':(b.name,),'ops':(('mul',.834),)}),
                       'test','test',divide,.001)
    fields,alt,dz=fields_for(table)
    program=compile_diagnostic(diag,table)
    shape=alt.shape if kind=='term_sum_3d' else alt.shape[1:]
    out=np.empty(shape,np.float32)
    evaluate_cpu(program,fields,alt,dz,out)
    expected=np.empty(shape,np.float32)
    f=np.float32
    for j in range(alt.shape[1]):
        for i in range(alt.shape[2]):
            column=f(0)
            for k in range(alt.shape[0] if kind!='term_sum_surface' else 1):
                v=f(fields[a.state_attr][k,j,i]+fields[b.state_attr][k,j,i])
                v=f(f(v*f(.38))/f(1.375))
                v=f(f(f(0)+v)+f(fields[b.state_attr][k,j,i]*f(.834)))
                if divide: v=f(v/alt[k,j,i])
                if kind=='term_sum_3d': expected[k,j,i]=v
                elif kind=='term_sum_surface': expected[j,i]=v
                else:
                    mass=f(f(f(1)/alt[k,j,i])*dz[k,j,i])
                    column=f(column+f(v*mass))
            if kind=='column_integral': expected[j,i]=f(column*f(.001))
    measured=ulp_table(out,expected)
    print(kind,divide,measured)
    assert measured==dict(max_ulp=0,n_nonzero=0,n=expected.size)
    native = np.empty_like(expected)
    evaluate(program, fields, alt, dz, native)
    native_words = ulp_table(native, expected)
    print("rust-host", kind, divide, native_words)
    assert native_words == dict(max_ulp=0, n_nonzero=0, n=expected.size)



_NC_SAFE = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_+.@-]*$")


@pytest.mark.parametrize("sets", [
    ["smoke"], ["dust"], ["gocart_primary"], ["gocart_simple"], ["cams_aq"],
    ["smoke", "gocart_primary"], ["smoke", "gocart_lite", "cams_aq"],
])
def test_history_names_every_process_array_once_and_nc_safe(sets, monkeypatch):
    """Every process array a run writes reaches history under one netCDF-safe
    name per row.  The smoke and GOCART lanes each added a process-output
    block to history_fields and the merge kept both: the smoke block wrote
    GOCART's template unexpanded ('EDUST{n}_DUST_1'), the per-domain wrfout
    writer refused that name, and every dust or sea-salt forecast stopped at
    its first history frame."""
    from woof import chem_table
    from woof.core.chem_context import chem_processes
    from woof.core.chem_history import history_fields
    from woof.core.chem_state import process_attr

    table = replace(chem_table.load_sets(sets), diagnostics=())
    processes = chem_processes(table) if table.processes else []
    state = SimpleNamespace()
    expected = 0
    for row in table.rows:
        setattr(state, row.state_attr, np.zeros((2, 3, 4), np.float32))
    for _key, module in processes:
        acted = module.rows(table)
        for alloc in getattr(module, "ALLOCATES", ()):
            n = len(acted) if alloc.shape.startswith("rows_") else 0
            plane = np.zeros(((n,) if n else ()) + (3, 4), np.float32)
            setattr(state, process_attr(alloc), plane)
            if alloc.output_name:
                expected += n if alloc.shape.startswith("rows_") else 1
    state.chem = SimpleNamespace(
        transported=table.transported, prescribed=table.prescribed,
        table=table, processes=processes)
    # A process that refreshes its outputs before each frame (live aerosol
    # optics, development branch f82d16b1c20) evaluates a CUDA kernel on the
    # real state; this host fake carries names only, so the refresh is
    # recorded here and must run once per such process per frame.
    refreshed = []
    expected_refresh = []
    for key, module in processes:
        if getattr(module, "prepare_history", None) is not None:
            expected_refresh.append(key)
            monkeypatch.setattr(
                module, "prepare_history",
                lambda st, _key=key: refreshed.append(_key))
    out = history_fields(state)
    assert refreshed == expected_refresh
    bad = [name for name in out if not _NC_SAFE.match(name)]
    assert bad == []
    rows_out = sum(1 for row in table.rows if row.output_name)
    assert len(out) == rows_out + expected
