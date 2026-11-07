"""Three-moment initial history and H_Z read the native closure without mutation."""
from types import SimpleNamespace

import numpy as np
from conftest import requires_gpu


def _case():
    import cupy as cp
    from test_p3_cuda_gpu import _tiny_column_case
    from woof.core import p3_device as PD

    host, nk, ncol = _tiny_column_case()
    fields = {k: cp.asarray(v) for k, v in host.items()}
    fields["qzi"] = 6 * fields["qi"] / cp.float32(np.pi * 300)
    fields["nc"].fill(1.e8)
    diag = {d: cp.zeros((nk, ncol), cp.float32) for d in PD.DIAG_SLOTS}
    surf = {s: cp.zeros(ncol, cp.float32) for s in PD.SURF_SLOTS}
    tables = PD.device_tables(three_moment=True)
    ws = PD.make_workspace(ncol, nk, three_moment=True)
    for it in range(1, 4):
        PD.run_p3_device(fields, diag, surf, workspace=ws, tables=tables,
                         dt=20., it=it, arm="sedlevels",
                         log_predictNc=True, log_3momentIce=True)
    return fields, diag, tables


@requires_gpu
def test_three_moment_diagnostic_matches_step_and_preserves_all_fields():
    import cupy as cp
    from woof.core import p3_device as PD

    fields, diag, tables = _case()
    inputs = {n: fields[n] for n in PD.REFLECTIVITY_FIELDS + ("qzi",)}
    before = {n: cp.asnumpy(v).tobytes() for n, v in fields.items()}
    out = cp.empty_like(fields["qi"])
    PD.run_p3_reflectivity(inputs, out, tables=tables)
    active = diag["zdbz"] > -98
    assert bool(active.any()) and bool((diag["zdbz"][active] > 0).any())
    # The current-state diagnostic uses final T, the step uses its entry T.
    assert float(cp.abs(out[active] - diag["zdbz"][active]).max()) < .05
    assert before == {n: cp.asnumpy(v).tobytes() for n, v in fields.items()}
    inputs["qzi"] = inputs["qzi"] * cp.float32(.01)
    changed = cp.empty_like(out)
    PD.run_p3_reflectivity(inputs, changed, tables=tables)
    assert not np.array_equal(cp.asnumpy(out).view(np.uint32),
                              cp.asnumpy(changed).view(np.uint32))


@requires_gpu
def test_operator_matches_production_final_diagnostics_on_identical_entry():
    import cupy as cp
    from woof.core import p3_device as PD

    fields, _, tables = _case()
    nk, ncol = fields["qi"].shape
    inputs = {n: fields[n] for n in PD.REFLECTIVITY_FIELDS + ("qzi",)}
    out = cp.empty_like(fields["qi"])
    PD.run_p3_reflectivity(inputs, out, tables=tables)
    # Invoke the unmodified production final routine on private carriers.
    # The only extra arithmetic initializes its entry-density companions
    # and converts the Registry's advected moment as the WRF wrapper does.
    source = PD.p3_source() + r'''
extern "C" __global__ void p3k_final_reference(P3_KERNEL_ARGS) {
    P3_TID;
    const int k = blockIdx.y;
    const float tm = r_pow(AT(F[F_PRES], k)*1.e-5f, P3_RD*P3_INV_CP);
    const float rho = AT(F[F_PRES], k)/(P3_RD*(AT(F[F_TH], k)*tm));
    AT(S[S_TMPARR1], k) = tm;
    AT(S[S_RHO], k) = rho;
    AT(S[S_INVRHO], k) = 1.f/rho;
    AT(S[S_RHOFACI], k) = r_pow(P3_RHOSUI/rho, .54f);
    const float ni_in = AT(F[F_NI], k);
    const float zi_in = AT(F[F_QZI], k);
    AT(F[F_QZI], k) = ni_in >= P3_QSMALL ? (zi_in*zi_in)/ni_in : 0.f;
    p3_final_level<1>(P3_PASS, k);
}
'''
    module = PD.p3_module(source=source)
    copies = {n: v.copy() for n, v in fields.items()}
    diag = {n: cp.zeros((nk, ncol), cp.float32) for n in PD.DIAG_SLOTS}
    surf = {n: cp.zeros(ncol, cp.float32) for n in PD.SURF_SLOTS}
    ws = PD.make_workspace(ncol, nk, three_moment=True)
    args = (PD._pointer_array([copies[n] for n in PD.FIELD_SLOTS]),
            ws.carrier_ptrs, ws.sedw_ptrs,
            PD._pointer_array([diag[n] for n in PD.DIAG_SLOTS]),
            PD._pointer_array([surf[n] for n in PD.SURF_SLOTS]),
            tables.pointers, ws.flags, np.int32(ncol), np.int32(nk),
            np.float32(20), np.int32(4), np.int32(1), np.int32(1),
            np.float32(1), np.float32(1))
    module.get_function("p3k_final_reference")((1, nk), (ncol,), args)
    np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32),
                                  cp.asnumpy(diag["zdbz"]).view(np.uint32))


@requires_gpu
def test_initial_history_and_da_use_three_moment_diagnostic():
    import cupy as cp
    from woof.core.state_reflectivity import diagnose
    from woof.da.obsop import simulated_reflectivity
    from woof.core import p3_device as PD

    fields, _, _ = _case()
    shape = (fields["qi"].shape[0], fields["qi"].shape[1], 1)
    slots = {}
    def scratch(shape_, name):
        return slots.setdefault(name, cp.empty(shape_, cp.float32))
    state = SimpleNamespace(p=fields["pres"].reshape(shape),
        thb=cp.zeros(shape[0], cp.float32), thp=fields["th"].reshape(shape),
        scratch=scratch, **{n: fields[n].reshape(shape)
        for n in ("qv", "qr", "nr", "qi", "qir", "ni", "qib", "qzi")})
    cfg = SimpleNamespace(mp_physics=53)
    initial = diagnose(state, cfg).copy()
    da = simulated_reflectivity(state, cfg)
    np.testing.assert_array_equal(cp.asnumpy(initial), cp.asnumpy(da))
    assert float(initial.max()) > 0
    # Explicit host-state H(x) shares the same CUDA closure.
    host = SimpleNamespace(**{n: cp.asnumpy(v) for n, v in vars(state).items()
                             if hasattr(v, "get")})
    np.testing.assert_array_equal(simulated_reflectivity(host, cfg), cp.asnumpy(initial))


@requires_gpu
def test_three_moment_clear_air_has_native_floor():
    import cupy as cp
    from woof.core import p3_device as PD
    from woof.da.obsop import CLEAR_AIR_FLOOR_DBZ

    fields, _, tables = _case()
    inputs = {n: fields[n].copy() for n in PD.REFLECTIVITY_FIELDS + ("qzi",)}
    for n in ("qr", "nr", "qi", "qir", "ni", "qib", "qzi"):
        inputs[n].fill(0)
    out = cp.empty_like(fields["qi"])
    PD.run_p3_reflectivity(inputs, out, tables=tables)
    np.testing.assert_array_equal(cp.asnumpy(out), np.float32(PD.CLEAR_AIR_DBZ))
    assert CLEAR_AIR_FLOOR_DBZ[53] == float(PD.CLEAR_AIR_DBZ)
