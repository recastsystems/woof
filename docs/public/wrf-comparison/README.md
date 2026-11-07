# WRF comparison evidence

These records compare WOOF with WRF component or forecast outputs. They are
code-verification evidence, not validation against observations. Each record
states its reference version, measured scope and failed or inconclusive gates.

- [Column evidence](mp28-column-evidence.md)
- [Matched idealized trajectory](mp28-matched-trajectory.md)
- [Short-window gate](mp28-shortwindow-gate.md)
- [Distribution gate](mp28-distribution-gate.md)
- [MAD-WRF real-input replay and measured cost](madwrf-port.md)

The three declared gate designs are retained byte for byte above their
`MEASUREMENTS` markers. Dated corrections below those markers distinguish
historical assumptions from later findings. The old `validation/` paths remain
as link stubs so retained receipts are not rewritten.
