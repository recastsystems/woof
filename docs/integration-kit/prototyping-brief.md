# Client design and integration notes

These notes describe implementation constraints and evaluation criteria for applications built around WOOF-ArWen 2.7. The [integration guide](integration-guide.md), installed capability catalogs, and [example client](examples/client.py) define the supported interface boundary. Application-specific design work does not introduce additional simulation APIs or imply that an unavailable engine capability exists.

## Application workflows

The desktop organizes its interface into Explore, Create forecast, and My forecasts. Independent clients can use a different organization while retaining the same distinctions between source data, an editable forecast setup, and a specific run's results.

| Workflow | Required state and behavior |
| --- | --- |
| Explore published weather | Source, initialization, member, product, and exact valid UTC come from available data. Playback selects actual frames within an explicit UTC range. |
| Create a forecast | The review identifies the source, area or cyclone target, duration, cadence, physics, computer, resource estimate, and output location. Execution uses the exact reviewed plan and configuration. |
| Inspect a saved or running forecast | The viewer remains bound to its selected run, domain, field, and time. Reconnecting reads durable state; opening or closing the client does not start or stop the forecast. |
| Request full analysis | Soundings and three-dimensional fields require an explicit analysis preparation for the selected frame. Compact two-dimensional map data is insufficient for a vertical profile. |

Source coverage, archive availability, forecast lead times, and member sets are capability data. A client must surface an unavailable selection and its reason rather than substituting a different source, date, or computer. Ordinary HRRR uses the same discovery and review boundaries as other supported sources.

## Interface boundaries and state ownership

Discovery uses `woof sources --json`, `woof run-plan --catalog`, and `woof run-plan --physics-profiles`. Plan review uses `woof run-plan PLAN.json --resolve` and `--estimate`; execution is the separate `woof run-plan PLAN.json` operation. The [prepared plan example](examples/prepared-plan.json) wraps an existing experiment TOML. Argument arrays, schema validation, exit-code checks, and explicit error handling are required at the process boundary.

Domain fitting, memory estimates, source preparation, physics compatibility, and simulation remain engine responsibilities. A client can present their results without maintaining a second implementation of those decisions. The desktop's `arwen.companion-handoff.v1` bridge is coupled to its release; independent applications should use the documented CLI boundaries rather than assume that bridge is a stable service API.

Draft state and results-viewer state have separate lifetimes. Changing a source, target, configuration, or plan invalidates the corresponding review. It does not replace a run already open in the viewer. Run manifests, request IDs, session and target identities, and configuration checksums retain their authority across reconnects.

## Asynchronous operations and cancellation

Queries, preparation, and transfers need visible pending and failure states. Responses remain bound to the request, selected target, and selection generation that produced them. Cancellation or a later selection must prevent a stale response from replacing the current view.

Cancelling a client query or transfer is distinct from stopping a forecast. Stop, resume, and launch are explicit operations against the intended run or reviewed plan. An unconfirmed launch acknowledgement must not trigger an automatic duplicate launch; the client should inspect the intended job state and explain the uncertainty.

A displayed frame retains its own timestamp while a replacement is prepared. Pausing preserves the selected UTC as new output arrives. Prefetch stays bounded, and following a moving domain is a separate camera choice. The map must not silently jump to another run or time because a background status update arrived.

## Scientific data and presentation

Transferred native data retains its producing run, domain, sequence, exact valid UTC, source authority, and geometry. Readers validate receipts, member checksums, path containment, and lease identity before opening the data. A shared cache lease remains held while any reader, map, or frame clone references the cached files; an evicted frame must be requested again.

Maps preserve the native product's units, projection, scale, color table, legend, contours, and wind barbs. A moved domain may have different geometry at each time. A map rendered as a small inset within an oversized empty plot is a layout defect, not an acceptable substitute for fitting its intended frame.

The current desktop uses Rust egui with native map rendering and a visual style based on pale sky and water colors, light surfaces, and clear controls. Other clients can choose a different style. Useful design criteria include readable labels, accessible contrast, keyboard operation, contextual controls, and sufficient space for the scientific map at both large and compact window sizes. Custom visual effects should have a defined rendering and asset implementation.

## Prototype boundaries

A disconnected prototype can exercise navigation, layout, and interaction using explicitly labeled fixtures. Mock weather, simulated progress, placeholder responses, and inactive execution controls must be distinguishable from an engine-connected workflow. Screenshots and performance figures should identify their source and measurement conditions. A prototype is not evidence of a completed forecast, available source data, or operational performance.

Each placeholder interaction should identify the documented call or missing capability required for integration. Private handoffs, node profiles, credentials, SSH material, and developer-machine paths do not belong in a distributed prototype. Scientific values and geometry require validation in addition to visual inspection.

## Evaluation criteria

Client validation should cover compact windows, large collections of subhourly frames, unavailable fields, slow first-frame preparation, moved domains, disconnected computers, changes after review, empty run lists, cache eviction, and incomplete historical coverage. Each case needs a clear explanation of the current state and an appropriate continuation or retry.

Design review records should include the interaction flow, its mapping to documented interfaces, asset and rendering requirements, and outstanding limitations. Visual preferences, usability defects, missing engine capabilities, and scientific correctness issues should be recorded separately so that each can be assessed on its own evidence.
