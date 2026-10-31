"""Write only runtime-mutated module scalars between column phases.

The native driver's reachable call graph changes these six scalar members.
Local arrays and module tables retain their descriptors between calls. Setup
still initializes the complete context, and a device copy initializes the separate collection output context. This lets the compiler discard unused local copies.
"""
MUTATED = ("ibinhlmlr", "ibinhmlr", "iehlw", "ipelec", "native_error", "renucfrac")


def lower(entry):
    original = "context.globals=column;"
    assert entry.count(original) == 1
    stores = "\n".join(f"context.globals.{name}=column.{name};" for name in MUTATED)
    entry = entry.replace(original, "if constexpr(Phase == 0) { context.globals=column; } else {\n" + stores + "\n}")

    old = "output[workspace_column].globals=column; output[workspace_column].driver=saved_driver;"
    assert entry.count(old) == 1
    stores = "\n".join(f"output[workspace_column].globals.{name}=column.{name};" for name in MUTATED)
    return entry.replace(old, stores)
