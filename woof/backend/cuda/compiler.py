"""Selected backend's source compiler, preserving historical import paths.

A name this module cannot serve (no local GPU, no backend) answers
``AttributeError`` as PEP 562 requires, so ``getattr(module, name, default)``
and module walks work; see ``woof.backend.cuda.__init__``.
"""
from woof.backend import array_module


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        compiler = array_module().cuda.compiler
    except Exception as error:
        raise AttributeError(f"woof.backend.cuda.compiler.{name}: {error}") from error
    return getattr(compiler, name)
