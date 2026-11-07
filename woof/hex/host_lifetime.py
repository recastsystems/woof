"""Release host-only inputs once all rank uploads and forcing setup passed."""
from types import SimpleNamespace


def release_preparation(host):
    prepared = host.get("prepared")
    if prepared is not None:
        # Keep the seal's small receipt for final run identity, not its mesh,
        # initialized state, terrain metrics and reference profiles.
        host["prepared"] = SimpleNamespace(expected_execution_fingerprint=
            getattr(prepared, "expected_execution_fingerprint", None))
    for name in ("constructor_values", "gwdo_host", "prepared_fingerprint_seal"):
        host.pop(name, None)
