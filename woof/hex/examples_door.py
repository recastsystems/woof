"""`woof hex examples`: where the runnable examples are, and a copy of them.

THE BREAKAGE THIS PREVENTS (0.6.0, a fresh install from the published files): the README, FIRST-HOUR and the
differentiable pages pointed at `examples/differentiable/` (run-32gb.sh, run-96gb.sh, own_measure.py,
expected.json), and `pip install` placed none of them: they were in the source archive and the repository only.
The wheel now carries them under `woof/hex/examples/` (setup.py copies `examples/differentiable/` there when it
builds), and this command prints the folder or copies it.
"""

from __future__ import annotations

import argparse
import filecmp
from pathlib import Path
import shutil
import sys

#: The example sets a wheel carries, as folder names under ``woof/hex/examples``.
SETS = ("differentiable",)


def examples_root() -> Path | None:
    """The folder holding the example sets: the installed package's, else a source checkout's."""
    here = Path(__file__).resolve().parent
    for root in (here / "examples", here.parents[1] / "examples"):
        if all((root / name / "README.md").is_file() for name in SETS):
            return root
    return None


def example_files(root: Path) -> list[Path]:
    return sorted(path for name in SETS for path in (root / name).rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts)


def run_examples(args: argparse.Namespace) -> int:
    root = examples_root()
    if root is None:
        print("woof hex examples: this install holds no examples folder (woof/hex/examples). They are in the "
              "release's source archive and in the repository under tree/examples/differentiable/.", file=sys.stderr)
        return 1
    files = example_files(root)
    if args.copy is None:
        print(f"examples: {root}")
        for path in files:
            print(f"  {path.relative_to(root).as_posix()}")
        first = root / "differentiable" / "run-32gb.sh"
        print(f"run one where it is:  bash {first} OUT")
        print("or copy them first:   woof hex examples --copy DIR")
        return 0
    target = Path(args.copy).expanduser().resolve()
    clash = [path for path in files
             if (target / path.relative_to(root)).is_file()
             and not filecmp.cmp(path, target / path.relative_to(root), shallow=False)]
    if clash and not args.force:
        names = ", ".join(str(path.relative_to(root)) for path in clash)
        print(f"woof hex examples: {target} already holds different files ({names}); nothing was copied. "
              f"Name an empty folder, or pass --force to replace them.", file=sys.stderr)
        return 2
    for path in files:
        out = target / path.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out)
        if path.suffix == ".sh":
            out.chmod(0o755)
    print(f"copied {len(files)} files to {target}")
    print(f"start with:  bash {target / 'differentiable' / 'run-32gb.sh'} OUT   (a 32 GB card; README.md beside it)")
    return 0


def add_examples_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "examples",
        help="print where the runnable examples are, or copy them to a folder",
        description=(
            "The runnable examples this install carries (the differentiable engine's scripts, with the numbers "
            "to expect). With no option, print their folder and files; with --copy DIR, copy them to DIR."
        ),
    )
    parser.add_argument("--copy", metavar="DIR", default=None,
                        help="copy the examples to DIR (DIR/differentiable/...); an existing different file stops it")
    parser.add_argument("--force", action="store_true", help="with --copy, replace files that differ")
    parser.set_defaults(handler=run_examples)
