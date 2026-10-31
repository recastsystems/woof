"""U-MPASPHYS-1: slot refusals keep their message and exception type, immediate or deferred."""
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - CPU tier
    pytest.skip("no CUDA device", allow_module_level=True)

from woof.hex.mpasphys.deferred_checks_v842 import DeferredChecks, collecting, defer


def test_immediate_outside_a_forecast():
    defer(cp.zeros(4, dtype=cp.int32), "never")
    with pytest.raises(ValueError, match="line 7"):
        defer(cp.array([0, 7, 3], dtype=cp.int32), "line {value}", error=ValueError)


def test_deferred_inside_a_forecast_raises_at_flush_or_finish():
    checks = DeferredChecks(cp)
    with collecting(checks):
        defer(cp.zeros(3, dtype=cp.int32), "clean")
        checks.flush(1)
        flag = cp.array([0, 5, 2], dtype=cp.int32)
        defer(flag, lambda v: f"first {v}", reduce="first", error=RuntimeError)
        flag[...] = 0          # later writes do not change the captured word
        checks.flush(2)
        with pytest.raises(RuntimeError, match=r"first 5 \(deferred device check, step 2\)"):
            checks.finish()
    assert checks.flushed == 2 and checks.words == 2


def test_refusal_is_read_before_output_and_wins_over_a_later_error():
    """The RRFS dycore validation word (SPEED-2 item 1): read before output, and named first when a later error unwinds."""
    from contextlib import ExitStack
    checks = DeferredChecks(cp)
    with collecting(checks):
        defer(cp.array([1], dtype=cp.int32), "dycore refused", error=FloatingPointError)
        checks.flush(3)
        with pytest.raises(FloatingPointError, match=r"dycore refused \(deferred device check, step 3\)"):
            checks.before_output()
    checks = DeferredChecks(cp)
    with pytest.raises(FloatingPointError, match=r"dycore refused \(deferred device check, step 4\)") as caught:
        with ExitStack() as stack:
            stack.enter_context(collecting(checks))
            stack.push(checks.prefer_earlier)
            defer(cp.array([1], dtype=cp.int32), "dycore refused", error=FloatingPointError)
            checks.flush(4)
            raise RuntimeError("a later slot check on the refused state")
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_output_flushes_checks_added_after_the_step_flush():
    checks = DeferredChecks(cp)
    with collecting(checks):
        checks.flush(3)
        flag = cp.array([7], dtype=cp.int64)
        defer(flag, "publication flag {value}", error=FloatingPointError)
        flag[...] = 0  # a one-word int64 snapshot must not alias this buffer
        wrote_output = False
        with pytest.raises(FloatingPointError, match=r"publication flag 7 .*step 3"):
            checks.before_output()
            wrote_output = True
        assert not wrote_output


def test_refusal_deadline_even_when_the_host_outruns_the_device():
    """An unfinished event must be waited at the next flush, even without new flags."""
    class UnreportedEvent:
        def __init__(self, event):
            self.event, self.waited = event, False

        @property
        def done(self):
            return self.waited and self.event.done

        def synchronize(self):
            self.event.synchronize()
            self.waited = True

    checks = DeferredChecks(cp)
    with collecting(checks):
        defer(cp.array([9], dtype=cp.int32), "bad flag {value}", error=ValueError)
        checks.flush(4)
        event, buffers, messages, step = checks._inflight[0]
        delayed = UnreportedEvent(event)
        checks._inflight[0] = delayed, buffers, messages, step
        with pytest.raises(ValueError, match=r"bad flag 9 .*step 4") as caught:
            checks.flush(5)
        assert delayed.waited
        assert caught.value.deferred_check_step == 4


def test_ready_checks_do_not_synchronize_at_plain_or_output_boundaries():
    class ReadyEvent:
        done = True

        def synchronize(self):
            raise AssertionError("ready checks must not synchronize")

    checks = DeferredChecks(cp)
    with collecting(checks):
        defer(cp.zeros(1, dtype=cp.int32), "clean")
        checks.flush(1)
        event, buffers, messages, step = checks._inflight[0]
        event.synchronize()  # explicitly establish readiness for this test
        checks._inflight[0] = ReadyEvent(), buffers, messages, step
        checks.flush(2)
        checks.before_output()
    assert not checks._inflight
