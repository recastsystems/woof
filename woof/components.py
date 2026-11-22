"""The component door: ``woof hex``.

The regional engine is the core of this distribution.  The hex model
(``woof.hex``, an MPAS-style unstructured-mesh port, preview) rides with it
as a subpackage, regional and global (``woof hex global``).  It keeps its own
complete command-line parser, so the front door hands the rest of the command
line to it unchanged instead of re-declaring its options here.

The spectral global model (``woof global``) was retired in WOOF 1.1.1; the
word stays only to say so and to name the global model that replaced it.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys

#: word -> (entry module, entry function, label, one-line help)
COMPONENTS = {
    "hex": ("woof.hex.cli", "main", "WOOF-HEX-A engine (preview)",
            "WOOF-HEX-A, the MPAS engine (hexagonal mesh), preview: woof hex --help"),
}

#: retired command words -> what to run instead
RETIRED = {
    "global": ("the spectral global model was retired in WOOF 1.1.1; global forecasts "
               "run on the hex model: woof hex global --help"),
}


def component_available(word: str) -> bool:
    module = COMPONENTS[word][0]
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def register_component_doors(sub) -> None:
    """List each component in the front door's help; dispatch happens earlier."""
    for word, (_module, _func, _label, text) in COMPONENTS.items():
        door = sub.add_parser(word, help=text, add_help=False)
        # Every word after the component's name goes to the component's own
        # parser (dispatch_component runs before this one), so this entry
        # defines no options of its own: a forwarding door, like `woof obs`.
        # (1.1.2) the docs-flag test held `woof hex ...` commands to this empty parser
        door._forward_argv = True


def dispatch_component(tokens: list[str]) -> int | None:
    """Run ``tokens`` through a component when its first word names one.

    Returns the component's exit code, or None when the words are not a
    component command and the regional parser should take them.
    """
    if tokens and tokens[0] in RETIRED:
        print(f"woof {tokens[0]}: {RETIRED[tokens[0]]}", file=sys.stderr)
        return 2
    if not tokens or tokens[0] not in COMPONENTS:
        return None
    word = tokens[0]
    module_name, func_name, label, _text = COMPONENTS[word]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and module_name.startswith(error.name):
            print(f"woof {word}: the {label} is not part of this install "
                  f"({module_name} is missing). Reinstall recast-woof.",
                  file=sys.stderr)
            return 2
        raise
    entry = getattr(module, func_name)
    print(f"woof {word}: the {label}; its config and checkpoint formats may "
          "change in a minor release.", file=sys.stderr)
    prog = f"woof {word}"
    saved = sys.argv[0]
    sys.argv[0] = prog
    try:
        result = entry(tokens[1:])
    except SystemExit as stop:
        code = stop.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv[0] = saved
    return 0 if result is None else int(result)
