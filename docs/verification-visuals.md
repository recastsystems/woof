# Observation verification

Completed runs queue observation verification through a detached CPU worker.
The foreground finish hook has an enforced two second cap, including queue
metadata and process creation: those operations run in a daemon launcher and
the caller uses a two second monotonic wait deadline. Thread setup and OS
scheduling can add overhead; this is not a real-time process deadline.
It performs no synchronous native or
public-source commands. The worker preserves compact inputs and scores only
already cached reference, station and radar inputs. Native reference lookup
uses `--offline`; artifact resolution cannot download a replacement binary.
Uncached observations remain pending for the explicit online arrival command.
The run's `verification.json` lists every discovered valid hour, native receipts
and pending reasons. Source failures leave the physical forecast verdict unchanged.

The finish hook itself writes no console output, so a blocked downstream stdout
pipe cannot trap a daemon printer at interpreter shutdown. After the hook has
returned, the run capsule prints one `verification:` line to stderr (see below);
a closed stream does not fail the finished run.

`verification-background.json` records the queue token, worker PID, local-only
mode, phase and retry command. `verification-background.log` is the worker log.
A run-local `.keep` with schema `gpuwm.verification-retention.v1` protects raw
inputs until every required forecast input has been prepared as compact native
planes. Operators and sweepers must retain the run while that lease exists.
The worker or a successful explicit retry removes only its token-owned lease;
a pre-existing manual `.keep` is never changed. Preparation or launch failures
retain the lease and the retry command. A busy online retry cannot block the
foreground queue, and repeated finish hooks coalesce a queued or running job.
Declared history paths and writer hashes bind the expected hour coverage to
native compact-input lineage. Missing declared frames stay pending and retain
the lease. Old receipts cannot release a newer job's coverage. Additional
finish declarations extend the active job; its worker revisits changed coverage.
Dead or reused worker PIDs are detected by process birth metadata, and a stale
job can be queued again. An explicitly empty frame declaration creates no lease.
If local filesystem or process creation remains blocked beyond two seconds,
the hook returns an in-memory pending state that forecast callers may discard.
A durable marker or visible message is not guaranteed in that timeout path.
Retain histories until a compact-ready receipt exists, then rerun the arrival
command. An ordinary detached worker can be interrupted
by host shutdown; its lease remains for recovery.

Run the arrival command again when more observations have been published:

```sh
woof verify-visuals RUN_DIRECTORY
woof verify-visuals RUN_DIRECTORY --list-pending
```

Each source arrives independently. Public METAR reports feed temperature,
dewpoint and wind verification. MRMS composite reflectivity and hourly QPE feed
fractional skill scores at 20/35 dBZ and 1/5 mm/h. Station reports before the
75 minute archive posting budget stay provisional. Reflectivity and QPE have
15 and 80 minute budgets. Repeating the command refreshes provisional hours and
fills pending hours. `--refresh` also refetches completed hours. Native receipts
state missing fields, paired counts, observation coverage and thresholds.

Rust retains only the required verification planes under
`domain/verification_inputs/valid-day/`, with the source hash lineage in a
reusable request. Pending hours can therefore be rescored after full history
files have been removed. Each preparation command has a bounded timeout in
the detached worker. A
preparation failure states the missing input or native failure; finish that
preparation before deleting the remaining histories.

The native reference table supplies the reference model input from the run's
initialization cycle and forecast lead. `--reference NAME` selects a table row.
`--cycle YYYY-MM-DDTHH:MM:SSZ` supplies the cycle when older histories omit it.
Python never decodes weather fields, samples stations or computes a score.

Images and receipts use `RUN_DIRECTORY/domain/product/valid-day/`. Scorecards
sit under `verification`. Map overlays use the field scale by default;
`--station-mode error` uses observation minus forecast dots. Receipt bias uses
forecast minus observation. Every convention is recorded in the native output.
Missing artifacts invalidate reuse, while unchanged complete hours are reused.
Replacing the selected native verification binary invalidates scored images
and receipts even when their inputs are unchanged. Compact input retention
remains reusable across scoring or rendering rebuilds.
The retained forecast and reference pair is bound to the first input revision
and selected reference row. Removing original histories or reference GRIB
files does not require downloading the reference again. Quantities absent
from either forecast arm are marked unavailable and complete; they remain
visible as missing quantities rather than waiting for observations forever.

Archived point and field inputs can be scored after raw histories are removed:

```sh
woof verify-visuals RUN_DIRECTORY --cycle 2026-01-01T00:00:00Z --domain d01 \
  --point-arm WOOF=POINT_DIRECTORY --point-arm REFERENCE=REFERENCE_POINTS \
  --field-arm 'WOOF=FIELDS/run-f{hour:02d}.npz' \
  --field-arm 'REFERENCE=FIELDS/reference-f{hour:02d}.npz' \
  --grid FIELDS/latlon.npz --station-table STATIONS.json \
  --first-hour 1 --last-hour 18 --append-to REPORT_DIRECTORY
```

Point filenames are `fNN-pts-w0.points.json`. NPZ field aliases are `t2_k`,
`td2_k`, `wind_ms`, `refc_dbz` and `precip_1h_mm`; the native reader reports
absent fields. A `--manifest FILE.json` can supply a list of per-hour native
requests for other layouts. Paired models must share valid times and comparable
quantities. Station-only hours can be scored without field files.

`--append-to` appends native score rows to `STATIONS.md` and `MRMS.md`, retaining
their existing bytes and skipping repeated rows. It formats native results and
does not pool or recalculate scores. A receipt with no observed events reports
that status rather than claiming a winner.

Run plans accept `run_options.verify_visuals = false`; `woof go`, `woof sim` and
`woof run` accept `--no-verify-visuals`. `WOOF_VERIFY_VISUALS=0` disables the finish hook for direct
runner calls. OFF performs no verification filesystem, network or worker-spawn work and does
not change the run configuration or physics. The explicit arrival command
always performs the requested verification.

When it is on, a finished forecast prints one `verification:` line saying what the
hook did. The step is a detached CPU process and uses no card. It can end a little
after the forecast process exits; the line names its process and the switch. It
runs for a forecast whose hours are all still ahead of the clock too: it makes the
raw history's compact verification inputs and releases the `.keep` lease, and
those hours stay pending for the arrival command. `--no-verify-visuals` on
`woof go`, `woof sim` or `woof run` skips it, and then the raw history is not
made compact.

The installed native artifacts are `rw_verify`, `rw_compare`, `rw_asos` and
`rw_mrms`. Explicit binary overrides use `WOOF_RW_VERIFY`, `WOOF_RW_COMPARE`,
`WOOF_RW_ASOS` and `WOOF_RW_MRMS`. A missing or incompatible artifact is named
in the per-hour pending receipt with the command needed to retry.
