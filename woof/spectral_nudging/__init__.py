"""GPU spectral nudging and prepared regional forecast integration."""
from .config import NudgingConfig, NudgingOptions


def build_nudger(config, bindings, **kwargs):
    """Off reads no fields, imports no CUDA code and allocates no buffers."""
    if not config.enabled:
        return None
    from .gpu import SpectralNudger
    return SpectralNudger(config, bindings, **kwargs)


__all__ = ["NudgingConfig", "NudgingOptions", "build_nudger"]
