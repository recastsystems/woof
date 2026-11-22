# SPDX-License-Identifier: Apache-2.0
"""What the differentiable doors read from a plan package, made once per plan and kept, not made again by every command.

THE COST THIS REMOVES. A door command read the plan's ``plan.json`` whole several times before its runner started, and
again while it wrote its answer: the card family of the recording, the timestep, the allocation table (the field map's
padding search, the answer's start state, the wind fold) each parsed it on their own, and the field map's padding
search inflated every recorded start payload to compare it. On the 3 km storm example ``plan.json`` is 707,709,916
bytes: one read and parse took 7.4 s on one machine (2026-11-14, read 4.1 s, parse 3.35 s), the padding search
17.8 s, and a later ``sensitivity`` command spent 42.25 s before its own clock started (card run of 2026-11-14, read from
the command's own file times), on no receipt. Every one of those answers is a function of the plan's
files alone.

What is kept, per plan, under the door's cache (``plan-facts/``; override ``WOOF_HEX_AD_PLAN_FACTS``): the sha256 of
plan.json, the card family of its recorded images, its timestep, its allocation table (as plan.json holds it), the
padded copies the field map found, and the wind fold's reading of the plan. The key is the plan folder, the name,
size, inode and change times of every file in it (``ad_doors._plan_files_sum``: a write moves the change time, which
no user call can set back), and the door programs that make the facts (their sha256) with the package version. Any
change of key makes the facts again from the files: a kept fact is never taken for another plan, another file or
another build. A cache that cannot be read or written is not an error: the facts are made from the files, as before.

The same module keeps the sha256 of the operator's files (the kernel set, the scheme bundle, the images) under their
own stamps, for the answer check's key and the session's key, which summed them whole in every command.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

FACTS_ENV = "WOOF_HEX_AD_PLAN_FACTS"
FACTS_SCHEMA = "gpuwm-hex.ad-plan-facts/v1"
SUMS_SCHEMA = "gpuwm-hex.ad-file-sums/v1"
#: The door programs whose code makes a fact: a change of any of them makes every fact again.
_CODE_FILES = ("ad_doors.py", "ad_plan_facts.py", "ad/rows.py", "wind_pool_fold.py", "ad/startwinds.py")

_MEMO: dict[str, "PlanFacts"] = {}
_CODE_SUM: list[str] = []
_SUMS: dict[str, Any] = {}


def facts_dir() -> Path:
    declared = os.environ.get(FACTS_ENV)
    if declared:
        return Path(declared).expanduser().absolute()
    from . import assets as _assets
    return _assets.cache_root() / "ad" / "plan-facts"


def code_sum() -> str:
    """sha256 over the door programs that make the facts, with the package version (read once a process)."""
    if not _CODE_SUM:
        from . import __version__
        digest = hashlib.sha256(str(__version__).encode("utf-8"))
        here = Path(__file__).parent
        for name in _CODE_FILES:
            try:
                digest.update(name.encode("utf-8") + b"\0" + (here / name).read_bytes())
            except OSError:
                digest.update(name.encode("utf-8") + b"\0absent")
        _CODE_SUM.append(digest.hexdigest())
    return _CODE_SUM[0]


def _write_atomic(path: Path, value: Any) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(value, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def process_age_s() -> float | None:
    """Seconds since this process was started (its start in /proc against the boot clock), or None where the kernel
    does not say. The door's own clock (``wall_s``) starts when its context is built; this is what came before."""
    try:
        text = Path("/proc/self/stat").read_text(encoding="ascii", errors="replace")
        ticks = int(text[text.rindex(")") + 2:].split()[19])
        return max(0.0, time.clock_gettime(time.CLOCK_BOOTTIME) - ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class PlanFacts:
    """The kept facts of one plan package. ``how`` reads "kept" when this process took them from the cache, "made"
    when it read the plan's files, None before anything was asked."""

    def __init__(self, plan: Path):
        self.plan = Path(plan)
        self.how: str | None = None
        self._facts: dict | None = None
        self._key: dict | None = None
        self._wind: dict | None = None

    # ---------------------------------------------------------------------------------------------- the kept file
    def _path(self) -> Path:
        name = hashlib.sha256(str(self.plan.resolve()).encode("utf-8", "surrogateescape")).hexdigest()[:24]
        return facts_dir() / f"{name}.json"

    def _load(self) -> dict:
        """The facts kept for the plan's files as they are NOW: the files' stamps are read at every call (a few
        milliseconds), so a plan changed under a long-lived process (the Python session object) makes them again."""
        from . import ad_doors as _d
        key = {"schema": FACTS_SCHEMA, "plan": str(self.plan.resolve()),
               "files": _d._plan_files_sum(self.plan), "code": code_sum()}
        if self._facts is None or key != self._key:
            self._key, self._wind = key, None
            try:
                doc = json.loads(self._path().read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = None
            if isinstance(doc, dict) and doc.get("key") == self._key and isinstance(doc.get("facts"), dict):
                self._facts, self.how = doc["facts"], "kept"
            else:
                self._facts, self.how = {}, "made"
        return self._facts

    def _save(self) -> None:
        _write_atomic(self._path(), {"schema": FACTS_SCHEMA, "key": self._key, "facts": self._facts,
                                     "made_utc": time.strftime("%Y-%m-%d %H:%MZ", time.gmtime())})

    def _fact(self, name: str, make: Callable[[], Any]) -> Any:
        facts = self._load()
        if name not in facts:
            facts[name] = make()
            self.how = "made"
            self._save()
        return facts[name]

    def _from_doc(self) -> None:
        """Every fact plan.json alone gives, from ONE read of it; the parsed document is dropped at once (it holds
        about 8 GB of host memory at the 3 km box)."""
        facts = self._load()
        if "allocations" in facts and "state_xtime" in facts:
            return
        from .ad import rows as _rows
        shared = getattr(_rows, "plan_doc", None)   # the command's one parse, where this build shares it
        doc = shared(self.plan) if shared else json.loads((self.plan / "plan.json").read_text(encoding="utf-8"))
        facts["allocations"] = doc.get("allocations")
        facts["dt_seconds"] = (doc.get("physics") or {}).get("dt_seconds")
        facts["state_xtime"] = (doc.get("physics") or {}).get("state_xtime")   # "" when the plan names none
        if facts["state_xtime"] is None:
            facts["state_xtime"] = ""
        try:   # the wind fold's reading where this build has hexcore.wind_pool_fold; a plan it cannot read keeps none
            from . import wind_pool_fold
            info = dict(wind_pool_fold.plan_info(self.plan, plan_doc=doc))
            info.pop("_al", None)
            info.pop("_R", None)
            facts["wind"] = json.loads(json.dumps(info))
        except Exception:   # noqa: BLE001  (the door's own call names the plan's fault when it needs the fold)
            facts.pop("wind", None)
        del doc
        self.how = "made"
        self._save()

    # ------------------------------------------------------------------------------------------------- the facts
    def sha256(self) -> str:
        """sha256 of plan.json."""
        from . import ad_doors as _d
        return self._fact("plan_sha256", lambda: _d._sha256_file(self.plan / "plan.json"))

    def allocations(self) -> list:
        """plan.json's allocation table, as it holds it."""
        self._from_doc()
        table = self._facts.get("allocations")
        if table is None:
            raise KeyError("allocations")
        return table

    def dt(self) -> float | None:
        """The timestep the plan was recorded with (plan.json physics.dt_seconds), when it names one."""
        try:
            self._from_doc()
        except (OSError, ValueError):
            return None
        value = self._facts.get("dt_seconds")
        return float(value) if value else None

    def state_xtime(self) -> str | None:
        """The clock the recording read before step 1 (plan.json physics.state_xtime), when the plan names one.

        THE BREAKAGE THIS PREVENTS: every answer of a sensitivity command parsed the whole plan.json (708 MB at the
        3 km box: 3.4 s and 4,271 MiB of host memory on one machine) to read this one date, in no named stage, so
        7 to 8 s of a target showed in no stage on the card (2026-11-14)."""
        self._from_doc()
        value = self._facts.get("state_xtime")
        return str(value) if value else None

    def arch(self) -> str | None:
        """The card family the plan's recorded forward images were built for (woof.hex.ad.rows.plan_arch)."""
        from .ad import rows as _rows
        return self._fact("arch", lambda: _rows.plan_arch(self.plan))

    def wind(self) -> dict:
        """hexcore.wind_pool_fold.plan_info of this plan, with the allocation table it reads the start words through."""
        from . import wind_pool_fold
        try:
            self._from_doc()
        except (OSError, ValueError):
            return wind_pool_fold.plan_info(self.plan)
        kept = self._facts.get("wind")
        if kept is None:   # not readable when the facts were made: the fold's own call says why
            return wind_pool_fold.plan_info(self.plan)
        if self._wind is None:   # one object a process: the fold keeps its operator matrix in it
            info = dict(kept)
            info["plan"] = str(self.plan)
            info["targets"] = {k: tuple(v) for k, v in kept["targets"].items()}
            info["_al"] = {int(a["id"]): a for a in self._facts["allocations"]}
            self._wind = info
        return self._wind

    def padded_copies(self) -> dict:
        """The field map's padding search, by root: "root:offset:shape:cells:edges" -> the allocations found."""
        return self._load().setdefault("padded_copies", {})

    @staticmethod
    def copies_key(root: int, offset: int, shape: list, mesh: dict) -> str:
        return f"{int(root)}:{int(offset)}:{'x'.join(str(int(x)) for x in shape)}:{mesh.get('cells')}:{mesh.get('edges')}"

    def keep_padded_copies(self) -> None:
        self.how = "made"
        self._save()


def plan_facts(plan: Path) -> PlanFacts:
    """The facts of ``plan`` (one object a plan folder a process)."""
    key = str(Path(plan).resolve())
    if key not in _MEMO:
        _MEMO[key] = PlanFacts(Path(plan))
    return _MEMO[key]


# ------------------------------------------------------------------------------------------- sums kept by stamp
def _stamp(path: Path) -> str | None:
    """A file's size, inode and change times, or a folder's files by name with theirs; None when it is not there."""
    try:
        if path.is_file():
            st = path.stat()
            return f"file\0{st.st_size}\0{st.st_ino}\0{st.st_mtime_ns}\0{st.st_ctime_ns}"
        if not path.is_dir():
            return None
        digest = hashlib.sha256()
        for sub in sorted(p for p in path.rglob("*") if p.is_file()):
            st = sub.stat()
            digest.update(f"{sub.relative_to(path).as_posix()}\0{st.st_size}\0{st.st_ino}\0{st.st_mtime_ns}\0"
                          f"{st.st_ctime_ns}\n".encode("utf-8", "surrogateescape"))
        return "dir\0" + digest.hexdigest()
    except OSError:
        return None


def kept_sum(path: Path | None, make: Callable[[Path], str | None]) -> str | None:
    """``make(path)`` (the sha256 of a file's bytes or of a folder's files), kept under the path and its stamp: the
    files are read once, and again only after one of them was written, replaced, added or removed. THE COST THIS
    REMOVES: the answer check's key summed the operator's files (about a gigabyte of kernel images and scheme
    bundles at the 3 km box) in every command that took a kept verdict."""
    if path is None:
        return None
    path = Path(path)
    stamp = _stamp(path)
    if stamp is None:
        return make(path)
    key = hashlib.sha256(f"{path.resolve()}\0{stamp}\0{code_sum()}".encode("utf-8", "surrogateescape")).hexdigest()
    store = facts_dir() / "sums.json"
    if not _SUMS:
        try:
            doc = json.loads(store.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and doc.get("schema") == SUMS_SCHEMA and isinstance(doc.get("sums"), dict):
                _SUMS.update(doc["sums"])
        except (OSError, ValueError):
            pass
        _SUMS.setdefault("", None)   # read once a process
    if key in _SUMS:
        return _SUMS[key]
    value = make(path)
    if _stamp(path) == stamp:   # a file written while it was read is not kept
        _SUMS[key] = value
        _write_atomic(store, {"schema": SUMS_SCHEMA, "sums": {k: v for k, v in _SUMS.items() if k}})
    return value
