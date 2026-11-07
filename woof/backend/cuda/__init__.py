"""Selected GPU backend compatibility namespace.

``woof.backend.cuda.<name>`` is ``array_module().cuda.<name>`` for the
selected backend.  A module ``__getattr__`` (PEP 562) must answer a name it
cannot serve with ``AttributeError``: ``getattr(module, name, default)``,
``hasattr`` and every module walk (the doctor's build-hint sweep,
``pkgutil`` consumers, ``inspect``) swallow only that.  Raising the backend's
``RuntimeError`` here made ``getattr(woof.backend.cuda, "CARGO_BUILD_HINT",
None)`` fail on a host with no local GPU (GPUWM_NO_LOCAL_GPU=1) instead of
answering ``None``.  The backend's reason is kept in the message and chained.
"""
from woof.backend import array_module


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        cuda = array_module().cuda
    except Exception as error:
        raise AttributeError(f"woof.backend.cuda.{name}: {error}") from error
    return getattr(cuda, name)
