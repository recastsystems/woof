# Untimed RRFS host-read census

`launch.py` runs the real forecast CLI with CUPTI runtime/driver API callbacks and
Python call-site stacks. Run only under the authorized GPU pool. Set
`RRFS_AUDIT_CUPTI` to the local `libcupti.so` and `RRFS_AUDIT_OUT` to a JSON receipt;
set the standard run recipe's `LAUNCHER` to this directory's `launch.py`.
`table.py receipt.json` prints per-step, per-operation driver counts.

The callback sees copies and synchronizations inside compiled CuPy code, including
implicit scalar conversions and size queries such as `flatnonzero`. Runtime rows
classify memcpy direction using the CUDA memcpy-kind argument. Driver rows name
DtoH copies directly; generic `cuMemcpyAsync` rows are kept as unclassified copies
and can be cross-checked against runtime rows at the same Python stack. Both layers
are recorded independently. **Never add runtime and driver counts together.**
`blocking` identifies a synchronous D2H API or a synchronize API. An asynchronous
D2H followed by stream synchronization is a blocking host read; the separate copy
and synchronization rows show both operations. The flag does not measure pageable
staging latency. Event-query calls do not block and are excluded.

The Python section counts explicit CuPy `item`, `get`, `asnumpy` and synchronize
calls. These can nest and are not independent copies. Numeric-slot calls such as
`int(cupy_scalar)` are attributed from the active Python CALL instruction at the
D2H callback. Their copies are counted even if source/position metadata is absent.
Stacks identify their calling expressions. Launch APIs are counted per operation;
Python `_launch` dispatch and `from_mpas` builders are counted separately.

Initialization uses `step: null`; forecast frames provide step numbers, and output
worker frames provide their capture step. Background calls without such frames
remain unassigned. Inspect `errors`, which must be empty. Audit failures are fatal.
The callback ABI is the stable prefix documented in NVIDIA's
[CUpti_CallbackData reference](https://docs.nvidia.com/cupti/13.2.0/api/structCUpti__CallbackData.html).
No GPU operations are issued from callbacks. The hook does not add synchronization,
but it adds host overhead, so **these runs cannot establish timing improvements**
or rule out waits that only occur when the host outruns the GPU without tracing.

`selftest.py` verifies five actual D2H copies from item, int, get, asnumpy and a
pinned asynchronous get, plus stream/device synchronization. It needs a small pool
lease. `test_schedules.py` checks cached schedule equality and invalidation on CPU.
`compare.py BASE_RUN TIP_RUN` checks all top-level NetCDF files and every variable,
rejecting missing files/variables/layout changes and comparing numeric bits and
text bytes. It needs no GPU.
