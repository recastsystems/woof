# Hex C1 events

The forecast door emits by default through `woof.timeline.ledger.emit`.
Early and retained step clocks use `Ledger.emit` to relay their original time.
The ledger lives in the engine (`woof.timeline`); hex provides no schema or
calculator of its own. An engine without that package (gpuwm 2.8.7 and
earlier) records nothing and prints nothing; an emitter that fails on an
engine that has it warns once per file without stopping the forecast. Use
`woof timeline` to calculate fractions on an engine that ships it.

`WOOF_TIME_LEDGER` selects the ledger file. Otherwise the destination is
`<out>/timeline.jsonl`. Early default events retain their original monotonic
and UTC clocks in memory until the driver creates its exclusive output
directory. A run refused before creating that directory has no default file.
Set an external ledger file to retain events from those refusals.

`WOOF_RUN_ID` and `WOOF_RENTAL_ID` come from the worker. Product is always
hex. The card field uses `WOOF_CARD`, then the first numeric entry of
`CUDA_VISIBLE_DEVICES`. A UUID or unknown assignment is null. No extra CUDA
query is made for telemetry.

| Stage | Boundary |
| --- | --- |
| prep.door | Request, admission, driver import and mesh binding |
| prep.host | Source and input checks plus host preparation |
| prep.cache_load | Each disk cache read and validation attempt, including misses |
| prep.device_upload | Device stack setup, including its kernel loads and transfers |
| forecast.construct | Door startup through first step start, including all preparation |
| forecast.first_step | First composite step through its existing stream synchronization |
| forecast.integrate | First step start through last completed step synchronization |
| forecast.output.capture | Snapshot capture, health/projection work and writer hand-off |
| forecast.output.history | Actual direct, background or fallback history write |
| forecast.output.wait | Waiting for all background frames and closing the writer |
| post.forecast | Last completed step through door receipt and final console output |
| finish | Door return, with returncode null if an exception leaves the door |

Construction, output and cache intervals can overlap. Integration includes
inter-step work and scheduled output between steps, as required by C1. Its
end clock is retained at the existing last-step synchronization and relayed
after the loop, so final history capture and validation belong to the tail.
No new synchronization is added.

The site series uses `prep.forecast_process` for the whole forecast subprocess.
That interval includes engine startup and postprocessing. It must not be
used as the integration numerator. The site's fetch, place, intermediate,
init, LBC, contract check, post and upload intervals may overlap while the
forecast process runs. Finish events carry a scope identifying the door or
the site; neither establishes when the machine was requested or released.

Changing source bytes invalidates the existing content-keyed prep cache. Each
build must warm its own cache before a paired warm measurement. The cache's
frozen source digest is updated to the telemetry version. Scientific identity
still requires a real full-forecast history/restart gate; CPU writer byte tests alone
do not establish it.
