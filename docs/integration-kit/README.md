# Integration kit (engine 2.7)

This kit documents the versioned CLI, plan documents, and durable run files used to integrate applications with WOOF-ArWen. It includes a Python subprocess client, interface examples, and capability catalogs. The desktop GUI and Rust TUI use the same forecast engine; the example client is an interface adapter, not a separate Python simulation API.

## Getting started

1. Read the [integration guide](integration-guide.md) for discovery, plan review, execution, and result inspection.
2. Run `python examples/client.py --help` for the example client's query and inspection commands.
3. Read the [client design and integration notes](prototyping-brief.md) for state ownership, asynchronous operations, scientific presentation, and prototype validation.

The examples query capabilities, review plans, and inspect existing runs. They do not start, stop, resume, or delete forecasts. An application's execution controls must use the reviewed configuration and an explicit operator action, as described in the guide.

## Files

- [Integration guide](integration-guide.md): supported boundaries, discovery, review, execution, reconnect, maps, and profiles.
- [Example client](examples/client.py): CLI JSON queries, plan review, and durable run inspection.
- [Prepared plan example](examples/prepared-plan.json): an envelope for an existing experiment TOML, with paths to replace before use.
- [Client checks](examples/test_client.py): schema and identity validation, monotonic events, and incomplete event tails.
- [Catalog exporter](examples/export_catalogs.py): regenerate capability snapshots from a selected runtime.
- `sources.json`, `physics-profiles.json`, and `catalog.json`: generated capability snapshots included in the packaged kit when the corresponding interface is available.
- [Client design and integration notes](prototyping-brief.md): implementation constraints and evaluation criteria for new interfaces.

## Runtime and catalog scope

Capability snapshots describe the runtime that generated them. Query the installed runtime again when constructing a real plan: availability depends on the source, requested time, input coverage, native components, and selected computer. Credential availability in a snapshot is not a statement about another installation. Exported machine-path placeholders such as `<USER_HOME>` are documentation values, not usable runtime paths.

An installed environment can invoke `python -m woof.cli` or its `woof` console command. Desktop packages that use an external Python environment select that interpreter according to their `INSTALL.md`. The self-contained Windows package includes `runtime/python.exe`. Credentials, SSH keys, geography, and forecast data remain in the operator's configured environment rather than in a distributable client.
