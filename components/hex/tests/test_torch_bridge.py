"""woof.hex.torch_bridge on the toy engine: the reverse, the forms, the models, the trainer.  Needs torch (extra)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from woof.hex.torch_bridge import proofs  # noqa: E402
from woof.hex.torch_bridge.autograd import run_window  # noqa: E402
from woof.hex.torch_bridge.engine import WindowSpec, to_engine, to_torch  # noqa: E402
from woof.hex.torch_bridge.losses import ScaledMSE  # noqa: E402
from woof.hex.torch_bridge.models import TableCorrection  # noqa: E402
from woof.hex.torch_bridge.trainer import Trainer  # noqa: E402


def test_toy_reverse_matches_central_differences():
    r = proofs.proof_reverse()
    assert r["rel"] < 1e-7, r


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_zero_model_is_the_engine_byte_for_byte(dtype):
    assert proofs.proof_zero(dtype=dtype)["bitwise_equal_all_steps"]


def test_exact_forms_match_differences_and_v0_is_measured():
    r = proofs.proof_grad(nonzero=0.05)
    for form in ("v1", "v0x"):
        assert r[form]["fd_rel_max"] < 1e-6, r
    assert r["v0x"]["rel_to_v1"] < 1e-12, r
    assert r["v0"]["rel_to_v1"] > 1e-6, r          # v0 is an approximation once the network reads its inputs
    exact = r["v0i_by_passes"][-1]                  # length - 1 passes: exact
    assert exact["rel_to_v1"] < 1e-12, r
    shrink = [p["rel_to_v1"] for p in r["v0i_by_passes"]]
    assert shrink == sorted(shrink, reverse=True), r


def test_v0_is_exact_at_iteration_zero():
    r = proofs.proof_v0_zero()
    assert r["grad_norm"] > 0 and r["rel_v0_to_v1"] < 1e-12, r


def test_keep_choices_agree():
    r = proofs.proof_keep()["max_abs_diff"]
    assert max(r.values()) < 1e-12, r


@pytest.mark.parametrize("form", ["v0", "v1"])
def test_table_recovers_planted_bias(form):
    r = proofs.proof_table(form=form, steps=150)
    assert r["rel_err_worst_field"] < 0.05, r


def test_column_net_learns_planted_term():
    r = proofs.proof_net(steps=80)
    first, last = r["held_out"][0][1], r["held_out"][-1][1]
    assert last < 0.3 * first, r


def test_multiscale_loss_trains():
    r = proofs.proof_net(steps=40, multiscale=True)
    assert r["held_out"][-1][1] < r["held_out"][0][1], r


def test_checkpoint_round_trip(tmp_path):
    cases = proofs._cases(n_cases=2, n_steps=8)
    eng = cases.model()
    model = proofs._to(TableCorrection(eng.nz, units=proofs.UNITS), eng)
    lines = []
    tr = Trainer(eng, model, cases, ScaledMSE(proofs.SCALES), form="v0", lr=0.05, log=lines.append,
                 ckpt_path=str(tmp_path / "w.pt"), ckpt_every=2)
    tr.fit(cases.windows([0], 4, 2), 3)
    for word in ("loss", "gnorm", "fwd", "rec", "rev", "net"):
        assert word in lines[0], lines[0]
    model2 = proofs._to(TableCorrection(eng.nz, units=proofs.UNITS), eng)
    tr2 = Trainer(eng, model2, cases, ScaledMSE(proofs.SCALES), form="v0", lr=0.05, log=lambda s: None)
    tr2.load(str(tmp_path / "w.pt"))
    assert tr2.step_no == 3 and torch.equal(model2.table, model.table)


def test_dlpack_shares_memory_cpu():
    a = np.zeros((4, 3))
    t = to_torch(a)
    t[1, 2] = 7.0
    assert a[1, 2] == 7.0
    b = to_engine(t, np)
    b[0, 0] = 3.0
    assert t[0, 0] == 3.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
def test_dlpack_shares_memory_cuda():
    cupy = pytest.importorskip("cupy")
    a = cupy.zeros((4, 3), dtype=cupy.float32)
    t = to_torch(a)
    assert t.is_cuda and t.data_ptr() == a.data.ptr
    b = to_engine(t, cupy)
    assert b.data.ptr == a.data.ptr


class _WholeReverseOnly:
    """The toy behind the interface the real runner gives today: a whole reverse, no pause at boundaries."""

    def __init__(self, e):
        self.e = e
        for k in ("xp", "dtype", "ncol", "nz", "slot_fields", "neighbours", "run_forward", "state",
                  "reverse_window"):
            setattr(self, k, getattr(e, k))


def test_whole_reverse_engine_runs_v0_and_v0i_and_refuses_v1():
    cases = proofs._cases()
    eng = _WholeReverseOnly(cases.model())
    model = proofs._net(eng.e, nonzero=0.05)
    win = WindowSpec(0, 0, 5)
    with pytest.raises(ValueError):
        run_window(eng, model, win, form="v1")
    grads = {}
    for form, passes in (("v0", 0), ("v0i", win.length - 1)):
        model.zero_grad(set_to_none=True)
        res = run_window(eng, model, win, form=form, passes=passes)
        sum(v.square().sum() for b in res.states.values() for v in b.values()).backward()
        grads[form] = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    res = run_window(cases.model(), model, win, form="v1")
    res.attach(sum(v.square().sum() for b in res.states.values() for v in b.values())).backward()
    g1 = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    assert float((grads["v0i"] - g1).norm() / g1.norm()) < 1e-12
    assert float((grads["v0"] - g1).norm() / g1.norm()) > 1e-6


class _FakeSession:
    """The bridge's Reverse over the toy: init / put / step / shadow with hexad's meaning (a table allocation's shadow
    slice k is step k's slot cotangent once the reverse has passed step k)."""

    def __init__(self, toy):
        self.toy, self.n = toy, toy._k
        self.shadows = {f"table:{f}": np.zeros((self.n, toy.nz, toy.ncol), toy.dtype) for f in toy.slot_fields}
        self.pending, self.calls = None, []
        self.state_shadows = {}

    def init(self, seed):
        self.toy._rev, self.pos, self.pending = None, self.n, seed
        for s in (*self.shadows.values(), *self.state_shadows.values()):
            s[...] = 0.0
        self.calls.append(("init", self.pos))

    def put(self, fields):
        from woof.hex.torch_bridge.engine import merge
        self.pending = merge(self.pending, fields)
        self.calls.append(("put", self.pos))

    def step(self):
        from woof.hex.torch_bridge.engine import merge
        for name, buf in self.state_shadows.items():      # adds made into a state allocation's shadow
            if name in proofs.PROG and buf.any():
                self.pending = merge(self.pending, {name: buf.reshape(self.toy.nz, -1).copy()})
            buf[...] = 0.0
        g = self.toy.reverse_step(self.pending)
        self.pending = None
        self.pos -= 1
        for f, a in g.items():
            self.shadows[f"table:{f}"][self.pos] = a

    def shadow(self, alloc, offset, nbytes):
        if alloc.startswith("state:"):
            buf = self.state_shadows.setdefault(alloc[6:], np.zeros(nbytes // np.dtype(self.toy.dtype).itemsize,
                                                                     self.toy.dtype))
            return buf
        flat = self.shadows[alloc].reshape(-1)
        i = offset // flat.itemsize
        return flat[i:i + nbytes // flat.itemsize]


def _hex_on_toy(cases):
    from woof.hex.torch_bridge.hex import HexEngine

    toy, replay = cases.model(), cases.model()
    hook = {}

    def forecast(window):
        toy.run_forward(window, hook["cb"], record=False)
        return toy.state()

    def open_session(window, tables):        # the session replays the forward with the table's words
        replay.run_forward(window, lambda k, s: {f: t[k] for f, t in tables.items()}, record=True)
        return _FakeSession(replay), {f: f"table:{f}" for f in tables}

    return HexEngine(forecast, open_session, ncol=toy.ncol, nz=toy.nz, xp=np, dtype=toy.dtype,
                     state_allocs={n: f"state:{n}" for n in ("v", "p", "ts")},
                     install=lambda cb, fields: hook.__setitem__("cb", cb), uninstall=lambda: hook.pop("cb"))


@pytest.mark.parametrize("form", ["v0", "v0i", "v0x", "v1"])
def test_hex_adapter_on_the_toy_gives_the_toys_gradient(form):
    cases = proofs._cases()
    eng = _hex_on_toy(cases)
    model = proofs._net(cases.model(), nonzero=0.05)
    win = WindowSpec(0, 1, 6)

    def grad(engine):
        model.zero_grad(set_to_none=True)
        tg = {b: {f: to_torch(v[f]) for f in proofs.PROG} for b, v in cases.targets(win).items()}
        res = run_window(engine, model, win, form=form, passes=win.length - 1)
        res.attach(ScaledMSE(proofs.SCALES)(res.states, tg)).backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()])

    assert torch.equal(grad(eng), grad(cases.model()))
