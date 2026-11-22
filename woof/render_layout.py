"""Where a rendered product PNG goes -- the one answer, for both engines.

Every picture a run drew used to land in ONE directory.  A three-nest
run of the rust catalog at hourly output puts five figures of files
there, of every product and every valid time, and the only way to find
one is to read filenames::

    out/case/png/arwen_wrf_20260520_18z_f000_d02-3km_composite_reflectivity.png
    out/case/png/arwen_wrf_20260520_18z_f000_d02-3km_2m_temperature.png
    out/case/png/arwen_wrf_20260520_18z_f000_d04-100m_composite_reflectivity.png
    ... 10,869 more

The layout below replaces that.  It is WOOF-ArWen's 2026-08-06 ruling -- case
folder, then domain, then product subfolders, organised AT RENDER TIME
rather than tidied afterwards -- with the reporter's timestamp request
slotted into it as the leaf grouping::

    <--out>/<domain-token>/<product>/<valid-day>/<filename>.png

Read as a sentence: *which nest, which chart, which day.*  The case
folder is ``--out`` itself, which every front door already sets per
case (``woof go`` gives it ``<case>/png``), so this module never
invents a case name -- it could not, and a case name in generic code is
a rule this project has paid for twice.

A nest that RETIRES and RE-ARMS lives more than once, and its lives are
separated one segment deeper::

    <--out>/<domain-token>/<episode>/<product>/<valid-day>/<file>.png

*which nest, which life, which chart, which day.*  The order above is
untouched -- domain still before product still before valid day -- and
the episode sits under the domain it is an episode OF, which is where
``woof.io.wrfout`` already files that nest's history
(``d05/episode-002/``).  Without it, two episodes of one nest at one
valid time -- the retiring episode's last frame and the re-armed
episode's ACTIVATION frame, which the history writer's own duplicate
guard exists to keep apart -- render to one delivered name, and the
second replaces the first with no failure and no warning.  A domain
that declares no ``retire``/``rearm`` has one life, reports episode
``0``, and files at exactly the three segments it always did.

Four properties are the contract, and each is pinned by a test in
``tests/test_render_layout.py``:

**It is the default.**  Not a flag, not an opt-in.  ``--layout flat``
exists only so a consumer written against the old directory has
somewhere to stand while it is updated, and it reproduces the v2.4.1
spelling byte for byte.  A correctness remedy that ships off is a
workaround; this one ships on.

**It is predictable.**  The path is a pure function of the three facts
in it, so a script can compute where a frame will be BEFORE it exists
and watch that one file, instead of globbing a directory and diffing
listings.  There is no adaptive bucketing, no "split when it gets big":
those cannot be predicted, which defeats the point.

**It never loses a file.**  Every segment has a defined value even when
the fact behind it cannot be read: an unidentifiable domain is
``native_grid`` (the spelling both engines already use), an unreadable
valid time is :data:`UNDATED`, an unparseable engine filename is
:data:`UNCLASSIFIED`.  A picture is always somewhere nameable, never
dropped and never left loose at the root.  Path LENGTH is part of that
promise on Windows, and it is answered TWICE.

:func:`delivered_name` answers the half that is ours: a frame filed
under ``<domain>/<product>/`` used to carry those same two tokens inside
its own filename, and a delivery measured on disk reached 310 characters
because of it.  The pair comes off at the organisation step -- the same
layout is then 226 characters under a typical case root -- and
:func:`engine_name` puts it back, so nothing is lost and no two frames
in a folder collapse onto one name.

:func:`fs_path` answers the half that is the caller's: a case root deep
enough still passes ``MAX_PATH``, the move into the tree fails, and the
picture is left flat at the root -- the ruling inverted for exactly the
products whose names are longest, while their shorter neighbours file
correctly and the directory looks almost right.

**One walker reads it.**  :func:`iter_rendered` is what every in-tree
consumer of a render directory uses, so a reader cannot be written that
sees only half the tree -- and it skips dot-directories, because the
early render's scratch is a dot-prefixed sibling of the pictures and a
naive recursive read would publish a half-finished run's temporaries.

The day is the day the frame is VALID, not the day its run was
initialised: a 21z cycle at f+06 files under the next morning, which is
the date a forecaster is looking for.
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
from pathlib import Path
from typing import NamedTuple

#: Windows' classic path ceiling.  A path of this length or longer is
#: rejected by the ordinary Win32 entry points -- ``mkdir``, ``replace``,
#: ``open`` and the directory walk alike -- with ERROR_PATH_NOT_FOUND,
#: which reads as "the folder is not there" rather than "the name is too
#: long".
_MAX_PATH = 260

#: The extended-length prefix.  A fully-qualified path wearing it is
#: handed to the filesystem verbatim, with no MAX_PATH ceiling and no
#: normalisation -- which is why :func:`fs_path` resolves the path first.
_LONG_PREFIX = "\\\\?\\"

#: Domain / product / valid-day subfolders.  The default.
NESTED = "nested"

#: Every PNG directly under ``--out``: what v2.4.1 and earlier wrote.
FLAT = "flat"

#: The vocabulary of ``woof render --layout``.
LAYOUTS = (NESTED, FLAT)

#: What ``woof render`` draws when nobody passes ``--products``
#: (``woof/render.py``'s own default), and so what a front door draws
#: when its caller names none.  The forecast doors read it through
#: :data:`woof.first_products.DEFAULT_RENDER_PRODUCTS`.  It is spelled
#: here because the preparation stage prints a ``woof sim`` line that
#: carries it, and the standalone preparation wheel stages this module
#: but not :mod:`woof.first_products`.
#:
#: ``all`` is every NAMED product the frames can draw: the renderer leaves
#: out the stored variables (``variables`` asks for those) and every window
#: the run's last frame does not close.
DEFAULT_RENDER_PRODUCTS = "all"

#: What ``--layout`` is when nobody says otherwise.
DEFAULT_LAYOUT = NESTED

#: The day segment for a frame whose valid time could not be read.
UNDATED = "undated"

#: The product segment for an engine output whose slug could not be read.
UNCLASSIFIED = "unclassified"

#: The domain segment for a file that proves no domain identity.  Spelled
#: as ``woof.render.NATIVE_GRID_SLUG`` and as the rust engine's
#: ``native_grid`` spell it -- one word for one fact.
NATIVE_GRID = "native_grid"

#: The head of the lifecycle-EPISODE segment, and the one place this
#: tree spells it.  ``woof.io.wrfout`` files an episodic domain's
#: history under ``d05/episode-002/`` and imports this to say so, so the
#: history tree and the delivered tree cannot drift into two spellings
#: of one fact.
EPISODE_PREFIX = "episode-"

#: An episode segment, as :func:`episode_segment` writes it.  Three
#: digits is the ZERO-PADDING width, not a ceiling: a slot re-armed a
#: thousand times writes four, and a reader that demanded exactly three
#: would stop recognising the tree at ``episode-1000``.
_EPISODE = re.compile(rf"^{EPISODE_PREFIX}(\d{{3,}})$")

#: ``YYYY-MM-DD`` at the head of a WRF ``Times`` record
#: (``1974-04-03_18:00:00``), its filename-safe form
#: (``1974-04-03_18-00-00``) or an ISO instant.  All three begin with the
#: date, which is the only part this module needs.
_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T_ ]|$)")

#: The HEAD of the rust engine's output filename: everything through the
#: forecast-hour marker.  It is the frame's own identity -- model, cycle
#: date, cycle hour, whole-hour lead -- and nothing in it is repeated by
#: any folder the layout builds, so it is exactly the part that survives
#: :func:`delivered_name`.
#:
#: ``model`` is non-greedy so the first eight-digit run is the date.
#:
#: The cycle hour is ONE OR TWO digits, and the one-digit form is not a
#: tolerance: it is what the engine writes.  ``store_render.rs`` formats
#: its ``cycle_utc: u8`` with a plain ``{}``, so a 06Z run is ``_6z_``
#: -- ten of the twenty-four cycle hours, GFS 00Z and 06Z among them.  A
#: parser that demanded two digits returned None for all of them, and
#: every frame of those runs was left flat under a front door printing
#: ``layout nested``.
#:
#: The LEAD is THREE OR MORE digits, for the same reason the episode
#: segment above is: three is the ZERO-PADDING WIDTH the engine formats
#: with, not a ceiling.  A run past 999 h writes ``f1000``, and a parser
#: that demanded exactly three stopped matching at that hour -- the
#: frame does not parse at all (``_ENGINE_NAME`` needs the underscore
#: right after the lead), so it is left flat at the render root and
#: :func:`woof.render_receipts` records it as :data:`UNCLASSIFIED`,
#: while :func:`delivered_name` hands the same name back unshortened.
#: ``woof.render``'s sibling parser already reads three-or-more, so
#: this is the two grammars agreeing on one number.
_HEAD = (r"(?:arwen|rustwx)_(?P<model>.+?)_(?P<date>\d{8})"
         r"_(?P<cycle>\d{1,2})z_f(?P<lead>\d{3,})")

#: The rust engine's output filename, as
#: ``rustwx-products``/``derived.rs`` formats it and
#: ``woof.render._rebrand_engine_output`` rebrands it::
#:
#:     arwen_<model>_<YYYYMMDD>_<H>z_f<NNN>_<domain-slug>_<product-slug>.png
#:
#: The tail is split into domain and product separately, because both
#: halves contain underscores and only their grammar tells them apart.
_ENGINE_NAME = re.compile(rf"^{_HEAD}_(?P<tail>.+)$")

#: The same grammar with the tail made OPTIONAL, which is what a
#: delivered name has: :func:`delivered_name` takes the tail off, so the
#: only thing left after ``f{NNN}`` is the exact-time suffix, or nothing
#: at all.  Spelled from the same ``_HEAD`` fragment as ``_ENGINE_NAME``
#: so the two cannot drift into disagreeing about what a frame is.
_DELIVERED_NAME = re.compile(rf"^{_HEAD}(?P<rest>.*)$")

#: A domain slug at the head of that tail: ``d02-3km``, ``d05-111m``, a
#: bare ``d02``, or the anonymous ``native_grid``.  The same three
#: degradation steps :func:`woof.render.domain_token` produces, which
#: are ``rw-wrfbatch::native_domain_slug``'s.
_TAIL = re.compile(
    r"^(?P<domain>native_grid|d\d{2}(?:-\d+(?:\.\d+)?(?:km|m))?)"
    r"_(?P<product>.+)$")

#: The engine's EXACT-TIME suffix, at the end of a product slug.
#:
#: ``f{NNN}`` in the filename counts whole hours, so two frames of one
#: product inside the same hour would collide.  The vendored engine
#: settles that itself: ``rusty-weather/src/render_all.rs`` builds
#:
#:     valid_{YYYY}{MM}{DD}_{HH}{MM}{SS}z_lead_{HHH}h{MM}m{SS}s
#:
#: with ``{lead_hours:03}`` (a MINIMUM width -- a run past 999 h writes
#: more digits) and two-digit lead minutes and seconds, and appends it.
#:
#: It identifies a FRAME, so it must not reach the product folder: a
#: parser that read it as part of the product name gave every frame a
#: folder of its own, which is the flat directory this module exists to
#: prevent wearing a nested costume, and on Windows pushed the path past
#: MAX_PATH so the frame was left flat outright.
#:
#: Anchored at the end and spelled out in full on purpose: a product
#: legitimately named ``valid_hours_since_analysis`` keeps its name.
_EXACT_TIME = re.compile(
    r"^(?P<product>.+)_valid_(?P<stamp>\d{8}_\d{6})z"
    r"_lead_\d{3,}h\d{2}m\d{2}s$")

#: A native SFIRE panel's file name (``rw-wrfbatch`` ``sfire::map_path``):
#: ``<product>_<YYYYMMDD>T<HHMMSS>.png``, the stamp being the frame's own
#: valid time, filed by the engine itself under
#: ``<domain>/<product>/<YYYYMMDD>/``.  Every fire product (perimeter,
#: rate of spread, heat flux, fuel, fire-grid terrain) is drawn per frame
#: under this name, the static fuel and terrain maps included.
NATIVE_PANEL = re.compile(
    r"(?P<product>[a-z][a-z0-9_]+)_(?P<stamp>[0-9]{8}T[0-9]{6})\.png")


def fs_path(path, *, descend: bool = False) -> str:
    """``path`` spelled so a filesystem call accepts it at ANY length.

    On Windows a path at or past :data:`_MAX_PATH` is refused by the
    ordinary API, and every one of this module's callers turns that
    refusal into the same degradation: the picture is left where the
    engine dropped it, flat at the render root.  That is WOOF-ArWen's
    2026-08-06 layout ruling inverted by nothing but arithmetic, and it
    is SELECTIVE -- it takes the longest product names first, so one
    frame of a set escapes the tree while its neighbours file correctly
    and the directory looks almost right.

    The remedy is the extended-length spelling, applied only when the
    ordinary one would fail: below the ceiling the caller's own path
    comes back unchanged, so error messages, receipts and anything a
    reader compares against stay in the spelling they typed.  Verbatim
    paths are not normalised by the OS, so the path is resolved to an
    absolute, ``..``-free, backslash form BEFORE the prefix goes on --
    a raw ``a/b/../c`` behind the prefix would name a directory called
    ``..``.

    ``descend=True`` is for a path that is about to be WALKED rather
    than opened.  A root's own length says nothing about its
    descendants', and the ceiling is enforced on the FULL path of each
    entry: a 200-character render directory holding a 269-character
    product file enumerates the directory happily and then reports the
    file as not a file.  A walk that started verbatim sees all of it.

    Not Windows, already prefixed, or short enough: unchanged.  This is
    a spelling, never a different file.
    """

    text = os.fspath(Path(path))
    if os.name != "nt" or text.startswith(_LONG_PREFIX):
        return text
    absolute = os.path.abspath(text)
    if not descend and len(absolute) < _MAX_PATH:
        return text
    if absolute.startswith("\\\\"):
        # A UNC share: \\server\share\... becomes \\?\UNC\server\share\...
        return _LONG_PREFIX + "UNC" + absolute[1:]
    return _LONG_PREFIX + absolute


def valid_day(stamp: str | None) -> str | None:
    """``1974-04-03`` from any stamp spelling this tree writes, or None.

    ``None`` is not an error: it means the caller must use
    :data:`UNDATED`, which :func:`place` does for it.  Guessing a day
    from a stamp that does not carry one would file a frame under a date
    it has no evidence for, and a wrong date is worse than an accurate
    ``undated`` because a reader believes it.
    """

    if not stamp:
        return None
    match = _DAY.match(str(stamp).strip())
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        # A syntactically well-formed date that does not exist
        # (``1974-02-31``) is no evidence at all.
        return None


def episode_segment(episode: int | None) -> str | None:
    """``episode-002`` for a lifecycle episode, ``None`` for no episode.

    The number is the lifecycle's own, as
    :func:`woof.core.nest_lifecycle.output_episode` reports it: a
    domain that declares ``retire``/``rearm`` reports the episode it is
    living (counting from 1), and a domain that declares neither reports
    ``0``.  ``0`` therefore means "this nest has one life", which is not
    an episode and gets no segment -- which is what keeps every run
    shipped so far filing at exactly the path it always did.

    Zero-padded to three digits so a listing sorts into episode order,
    and no wider than the number needs, so ``episode-1000`` is still
    that episode's folder rather than a truncation of it.
    """

    if episode is None:
        return None
    number = int(episode)
    if number <= 0:
        return None
    return f"{EPISODE_PREFIX}{number:03d}"


def episode_number(name) -> int | None:
    """The episode a folder name spells, or ``None`` if it spells none.

    The exact inverse of :func:`episode_segment`, and the reason both
    exist here: the render side learns which episode a history file
    belongs to by reading the folder the history writer filed it in, so
    a name written by one function is read back by the other rather
    than by a second copy of the grammar.

    ``episode-000`` answers ``None``.  It is what a lifecycle-free
    domain would spell if anything ever wrote it, and reading it as an
    episode would put a nest with one life under a segment that claims
    it had more.
    """

    if name is None:
        return None
    match = _EPISODE.match(str(name).strip())
    if match is None:
        return None
    number = int(match.group(1))
    return number if number > 0 else None


class HistoryFrame(NamedTuple):
    """One wrfout frame on disk: where it is, which nest, which life.

    ``grid`` is the ``dNN`` token off the filename (``None`` for a name
    that carries none), and ``episode`` is the lifecycle episode read
    back out of the folder the history writer filed it in -- the exact
    inverse of :func:`episode_segment`, so the writer's spelling and the
    reader's cannot drift into two.  ``None`` means the frame sits at
    the top level, which is where a nest with one life writes.
    """

    path: Path
    grid: str | None
    episode: int | None


#: The head of a wrfout history filename: ``wrfout_d02_1974-04-03_18_00_00``.
#: Only the grid token is read here; the valid time is carried by the
#: name's own sort order, which is why :func:`history_frames` orders on
#: the name rather than parsing it.
_HISTORY_NAME = re.compile(r"^wrfout_(?P<grid>d\d{2})_")


def history_frames(root) -> list["HistoryFrame"]:
    """Every wrfout history frame under ``root``, in time order.

    THE reader for a history directory, and the counterpart of
    :func:`iter_rendered` on the input side.  A domain that ATTACHES
    mid-run, and one that retires and re-arms, files its frames one
    segment deeper -- ``d05/episode-002/wrfout_d05_...`` -- exactly as
    :func:`episode_segment` spells it, because ``woof.io.wrfout``
    writes them there.  An enumerator that globbed the top level only
    saw none of those frames, and a run whose only frame producer was
    that nest reported that it published no history after writing a
    full one.

    Three things this does that a bare ``rglob`` does not.

    It SKIPS readiness JSON sidecars and the writer's in-flight temporaries (``wrfout*.tmp*``,
    ``woof.io.wrfout``'s own spelling) and anything that is not a
    regular file, so a half-written frame is never handed to a reader
    as though it were finished.

    It SKIPS every frame under a dot-prefixed folder, the rule
    :func:`woof.io.wrfout.is_wrfout_frame` makes: ``.quarantine`` holds
    swept frames and a head-bound attempt the forecast discarded
    (:data:`woof.prepared_domain_tree_forecast.STREAMED_ATTEMPT_PARENT`),
    whose frames carry the sealed forecast's names and are not its
    history.

    It orders on ``(name, episode)`` rather than on the path string.
    ``p.name`` already carries domain-then-valid-time, so it is the time
    order the caller's docstring promises, while a path-string sort puts
    every nested episode ahead of a top-level frame of an earlier hour.
    The episode is the tie-break because two episodes of one slot can
    publish the SAME valid time -- the retiring episode's last frame and
    the re-armed episode's activation frame -- and the writer's
    duplicate guard is keyed on the full path, so it does not catch that
    collision.
    """

    root = Path(root)
    walk_root = Path(fs_path(root, descend=True))
    if not walk_root.is_dir():
        return []
    found: list[HistoryFrame] = []
    for path in walk_root.rglob("wrfout_d*"):
        if ".tmp" in path.name or path.suffix.lower() == ".json" or not path.is_file():
            continue
        relative = path.relative_to(walk_root)
        if any(part.startswith(".") for part in relative.parts[:-1]):
            continue
        grid = _HISTORY_NAME.match(path.name)
        found.append(HistoryFrame(
            root / relative,
            grid.group("grid") if grid is not None else None,
            episode_number(path.parent.name)))
    return sorted(found, key=lambda frame: (frame.path.name,
                                            frame.episode or 0))


def product_dir(*, domain: str | None, product: str | None,
                day: str | None, episode: int | None = None) -> Path:
    """The relative directory one product frame belongs in.

    Relative on purpose: the root is the caller's ``--out``, and a
    function that joined it would be one that could be handed the wrong
    root without anyone noticing.

    ``episode`` EXTENDS the 2026-08-06 ruling rather than reordering it:
    domain still comes before product, which still comes before the
    valid day, and the episode sits under the domain it is an episode
    OF -- the same place ``woof.io.wrfout`` already puts it in the
    history tree.  Absent (the default, and every run that declares no
    ``retire``/``rearm``), the directory is the three segments it has
    always been, character for character.
    """

    directory = Path(domain or NATIVE_GRID)
    segment = episode_segment(episode)
    if segment is not None:
        directory = directory / segment
    return directory / (product or UNCLASSIFIED) / (day or UNDATED)


def place(root, *, domain: str | None, product: str | None,
          day: str | None, filename: str, episode: int | None = None,
          layout: str = DEFAULT_LAYOUT) -> Path:
    """The full path for one rendered PNG, under ``root``.

    ``layout=FLAT`` returns ``root / filename`` -- the v2.4.1 spelling,
    unchanged, so a consumer pinned to it keeps working while it is
    updated.  It has no folders to carry an episode and grows none:
    ``--layout flat`` is a compatibility spelling, and a flat directory
    that quietly gained a subdirectory would not be one.
    """

    root = Path(root)
    if layout == FLAT:
        return root / filename
    if layout != NESTED:
        raise ValueError(
            f"unknown render layout {layout!r}; choose from "
            f"{', '.join(LAYOUTS)}")
    return root / product_dir(domain=domain, product=product,
                              day=day, episode=episode) / filename


def _folder_tokens(name: str, *, domain: str | None,
                   product: str | None) -> tuple[str, str, str] | None:
    """``(head, repeated, rest)`` for one frame, or None if it does not fit.

    ``repeated`` is the exact ``<domain>_<product>`` string the two
    folders above the frame spell.  ``None`` means this name and these
    folders do not line up -- an engine name the grammar cannot read, or
    the accurate fallback where neither the caller's token nor the slug
    grammar could split the tail -- and every caller then leaves the
    name exactly as it found it rather than cutting a guess out of it.
    """

    if not domain or not product:
        return None
    stem = Path(name).stem
    match = _DELIVERED_NAME.match(stem)
    if match is None:
        return None
    return stem[:match.start("rest")], f"{domain}_{product}", match.group(
        "rest")


def delivered_name(name: str, *, domain: str | None,
                   product: str | None) -> str:
    """``name`` with the tokens its own folders already carry removed.

    The engine writes ``arwen_<model>_<date>_<cycle>z_f<NNN>_<domain>_
    <product>[_<exact-time>].png`` into one flat directory, where every
    token has to be there because nothing else tells two files apart.
    Filed into the layout, two of those tokens become the names of the
    folders the frame is sitting in, and the picture then carries them
    twice::

        .../d01-12km/var_geopotential_height_700hpa_38d0bbbc4b4b7e87/
            2026-08-20/arwen_wrf_20260820_0z_f000_d01-12km_var_
            geopotential_height_700hpa_38d0bbbc4b4b7e87_valid_...png

    That is not untidiness, it is a delivery defect.  A real one measured
    310 characters, and while :func:`fs_path` lets WOOF-ArWen write and read
    past MAX_PATH, it does nothing for the tools the delivery is OPENED
    with: Explorer, ``tar``, and the readers a recipient's own script
    imports all refuse the path.  The picture is filed correctly and
    cannot be opened, which is the same lost picture by a third route.

    So the repeated pair comes off HERE, at the Python organisation step,
    and not in the engine: the vendored crate stays byte-identical to its
    campaign builds, and ``--layout flat`` -- where the folders spell
    nothing and every token is essential -- keeps the v2.4.1 name
    byte for byte.

    What survives is the frame's own identity (model, cycle date, cycle
    hour, lead) and the engine's exact-time suffix, which no folder
    carries.  Names therefore stay collision-free: :func:`engine_name`
    rebuilds the engine's name from the delivered one and its two
    folders exactly, so two frames that differed before differ after.

    A name the grammar cannot read, or one whose folders do not spell
    what it carries, comes back unchanged.
    """

    tokens = _folder_tokens(name, domain=domain, product=product)
    if tokens is None:
        return Path(name).name
    head, repeated, rest = tokens
    if not rest.startswith(f"_{repeated}"):
        return Path(name).name
    return head + rest[len(repeated) + 1:] + Path(name).suffix


def engine_name(name: str, *, domain: str | None,
                product: str | None) -> str:
    """The engine's own filename, rebuilt from a delivered one.

    The exact inverse of :func:`delivered_name` given the same two
    folders, and the reason the shortening is safe to do at all: it is
    information-preserving, so a reader that wants the v2.4.1 spelling
    -- ``--pair``'s matching key is the one in this tree -- computes it
    instead of being handed a name it can no longer recognise.

    A name that already carries the pair is already the engine's, and
    comes back unchanged; so does one the grammar cannot read.
    """

    tokens = _folder_tokens(name, domain=domain, product=product)
    if tokens is None:
        return Path(name).name
    head, repeated, rest = tokens
    if rest.startswith(f"_{repeated}"):
        return Path(name).name
    return f"{head}_{repeated}{rest}{Path(name).suffix}"


def parse_engine_output(name: str, *, domain: str | None = None
                        ) -> tuple[str, str, str] | None:
    """``(domain, product, valid_day)`` for one rust-engine filename.

    ``None`` when the name is not one of the engine's at all -- the
    caller then leaves the file where the engine put it rather than
    filing it under a guess.

    ``domain`` is the token the caller read from the wrfout itself
    (:func:`woof.render.domain_token`).  It is a HINT for splitting the
    tail -- domain slugs and product slugs both contain underscores, so
    knowing one end tells you where the other begins -- and the fallback
    when the filename carries no parseable slug.  It does not override
    the filename's own slug, because the folder a picture lands in must
    be spelled the same way as the token inside its filename: a
    ``d02-3km`` file under a ``d02/`` folder is a reader asking which of
    the two is the lie, every time.  In a real render they are the same
    string; they differ only when one side could read less of the file
    than the other, and the more specific answer is the useful one.

    The day is the frame's VALID day: cycle date + cycle hour + lead
    hours.  A 21z run at f+06 is the next morning, and filing it under
    the initialisation date would put the interesting frames of every
    evening run in the wrong folder.
    """

    match = _ENGINE_NAME.match(Path(name).stem)
    if match is None:
        return None
    tail = match.group("tail")
    product: str | None = None
    if domain and tail.startswith(f"{domain}_"):
        product = tail[len(domain) + 1:]
    else:
        split = _TAIL.match(tail)
        if split is not None:
            domain = split.group("domain")
            product = split.group("product")
    if not product:
        # A tail neither the caller's token nor the slug grammar can
        # split.  The domain is still known (or accurately anonymous), and
        # the whole tail is a truthful, if ugly, product name -- better
        # than dropping the file at the root where it is invisible.
        product = tail
    # A sub-hourly frame carries the engine's exact-time suffix.  It
    # names the FRAME, so it comes off the product -- and it carries the
    # frame's own valid stamp, which is better evidence of the valid day
    # than cycle + whole-hour lead, the only thing available without it.
    exact = _EXACT_TIME.match(product)
    if exact is not None:
        product = exact.group("product")
        try:
            stamp = datetime.datetime.strptime(
                exact.group("stamp"), "%Y%m%d_%H%M%S")
        except ValueError:
            # A well-formed but impossible stamp is evidence of nothing,
            # and the same call `valid_day` makes for one.  The suffix
            # still comes off: the engine wrote it, and it is still not
            # the product's name.
            return (domain or NATIVE_GRID), product, UNDATED
        return (domain or NATIVE_GRID), product, stamp.date().isoformat()
    try:
        # The cycle hour is zero-padded HERE, not required of the name:
        # the engine writes ``6z``, and handing strptime a nine-digit
        # string would let ``%H`` steal a date digit.
        cycle = datetime.datetime.strptime(
            f"{match.group('date')}{int(match.group('cycle')):02d}",
            "%Y%m%d%H")
        valid = cycle + datetime.timedelta(hours=int(match.group("lead")))
    except ValueError:
        return (domain or NATIVE_GRID), product, UNDATED
    return (domain or NATIVE_GRID), product, valid.date().isoformat()


def engine_output_time(name: str) -> datetime.datetime | None:
    """The instant one engine output filename names, or ``None``.

    A TOTAL function: every name the grammar cannot read answers
    ``None``, and nothing here raises.  That is the whole point of it
    living in this module.  ``woof.render``'s series-with-context path
    re-reads the clock off every PNG the engine wrote, and a local
    helper that raised on an unreadable name killed the render AFTER the
    engine had already drawn the frames -- work paid for and then
    discarded.  A caller that gets ``None`` has one fact it can act on
    ("this is not one of the frames I asked for"), and it can say so by
    name instead of ending the run.

    The order of evidence is :func:`parse_engine_output`'s, spelled from
    the same two fragments so the two cannot drift: the engine's
    exact-time suffix first, because it carries the frame's own stamp,
    and cycle date plus cycle hour plus lead otherwise.  A lead past
    999 h is read in full (``f1000`` is 1000 hours, not 100), which is
    the same three-or-more-digit grammar ``_HEAD`` pins.

    The instant is naive UTC, as the engine writes it.  A native SFIRE
    panel (:data:`NATIVE_PANEL`) carries its valid time as
    ``YYYYMMDDTHHMMSS`` and is read from that stamp.
    """

    native = NATIVE_PANEL.fullmatch(Path(name).name)
    if native is not None:
        try:
            return datetime.datetime.strptime(
                native.group("stamp"), "%Y%m%dT%H%M%S")
        except ValueError:
            return None
    stem = Path(name).stem
    head = _DELIVERED_NAME.match(stem)
    if head is None:
        return None
    exact = _EXACT_TIME.match(stem)
    if exact is not None:
        try:
            return datetime.datetime.strptime(
                exact.group("stamp"), "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    try:
        # The cycle hour is zero-padded HERE, never required of the
        # name: the engine writes ``6z``.
        cycle = datetime.datetime.strptime(
            f"{head.group('date')}{int(head.group('cycle')):02d}",
            "%Y%m%d%H")
    except ValueError:
        return None
    return cycle + datetime.timedelta(hours=int(head.group("lead")))


def probe_delivery_root(root) -> str | None:
    """``None`` when a delivery root can take a file, else one sentence.

    Asked ONCE at plan review, before anything is drawn, and answering
    only the question that is actually knowable then: can this root take
    a file and give it back.  A root that cannot is a run whose every
    picture is lost, which is worth refusing on before the forecast
    rather than after it.

    The root is CREATED if it is not there, because that is what a
    render does with it a moment later and a probe that refused an
    absent directory would refuse every first run.

    It deliberately does NOT probe the deepest path the plan will
    reach.  A probe at t=0 does not predict an ACL, a full disk or a
    held handle at t plus forty minutes, and refusing a run on a
    prediction is refusing something that has not happened.  The depth
    is answered where it is real instead: :func:`fs_path` wears the
    extended-length spelling and :func:`deliver` reports what actually
    went wrong, once, with the OS's own words.
    """

    root = Path(root)
    probe = root / ".gpuwm-delivery-probe"
    try:
        Path(fs_path(root)).mkdir(parents=True, exist_ok=True)
        spelled = fs_path(probe)
        with open(spelled, "wb") as stream:
            stream.write(b"")
        os.unlink(spelled)
    except OSError as error:
        return (f"delivery root {root} cannot take a file ({error}); "
                "every picture this run draws would be lost -- name a "
                "writable directory with --out.")
    return None


def deliver(root, source, *, domain: str | None, product: str | None,
            day: str | None, filename: str | None = None,
            episode: int | None = None,
            layout: str = DEFAULT_LAYOUT) -> tuple[Path, str | None]:
    """File one drawn picture into the layout; ``(where it is, note)``.

    THE placement seam, and the reason this module has one: the
    contract above says a picture is always somewhere nameable, never
    dropped and never left loose at the root, and every caller that
    hand-rolled the move implemented a different half of it.

    ``domain``, ``product`` and ``day`` may each be ``None``: that is
    not an error, it is :func:`product_dir`'s defined degradation, so a
    frame whose product could not be read files under
    ``<domain>/unclassified/undated/`` and stays inside the tree a
    reader walks.

    The move is tried twice before anything degrades.  ``os.replace``
    first; on ``OSError`` a copy followed by an unlink, which is what a
    cross-device scratch and a briefly held handle both need.  Only
    when both fail does the picture stay where it was, and the returned
    path is then the source, so the caller's list never names a file
    that is not there.

    ``note`` is ``None`` when the ordinary route worked, and one
    sentence otherwise -- RETURNED rather than printed, so a caller can
    both show it and record it in the render receipt.  A degradation
    that only ever reached stderr is one the run's own summary cannot
    report.
    """

    source = Path(source)
    target = place(root, domain=domain, product=product, day=day,
                   filename=filename or source.name, episode=episode,
                   layout=layout)
    if target == source:
        return source, None
    spelled = fs_path(target)
    try:
        Path(spelled).parent.mkdir(parents=True, exist_ok=True)
        os.replace(fs_path(source), spelled)
        return target, None
    except OSError as error:
        first = error
    try:
        Path(spelled).parent.mkdir(parents=True, exist_ok=True)
        with open(fs_path(source), "rb") as reading, \
                open(spelled, "wb") as writing:
            shutil.copyfileobj(reading, writing)
        os.unlink(fs_path(source))
    except OSError as error:
        return source, (f"left flat, could not move into layout "
                        f"({first}) and the copy fell back to the same "
                        f"({error}): {source.name}")
    return target, (f"filed by copy, the move could not be done in "
                    f"place ({first}): {target.name}")


#: A render's working scratch sits beside its delivery as ``<delivery><SCRATCH_SUFFIX>/`` (``woof.render``
#: spells it from here), and holds working stores that carry stray PNGs of their own.
SCRATCH_SUFFIX = ".render-scratch"


def is_scratch_dir(name: str) -> bool:
    """A folder no reader lists pictures from: dot-prefixed temporaries and a render's working scratch."""

    return name.startswith(".") or name.endswith(SCRATCH_SUFFIX)


def iter_rendered(root) -> list[Path]:
    """Every rendered PNG under ``root``, flat layout or nested.

    THE reader for a render directory, and the reason there is only one:
    a consumer that globbed ``*.png`` saw nothing after this layout
    landed, and a consumer that recursed naively saw the early render's
    ``.first-products-scratch`` temporaries as though they had been
    published (and a page scanning a run folder counted the render's
    sibling ``<delivery>.render-scratch/`` as a grid).  Both mistakes
    are made once, here.

    Sorted by path so two directories walk in the same order, which is
    what ``--pair`` needs to line frames up.

    The walk goes through :func:`fs_path` and the results come back in
    the CALLER's spelling.  A reader that could not see as deep as the
    placement can write is the same lost picture by another route: the
    file would be correctly filed, invisible to the early-render
    publisher, and reported as "produced no picture".
    """

    root = Path(root)
    walk_root = Path(fs_path(root, descend=True))
    if not walk_root.is_dir():
        return []
    found = []
    for path in walk_root.rglob("*.png"):
        relative = path.relative_to(walk_root)
        # A run folder holds its delivery AND the delivery's sibling scratch (``png`` beside
        # ``png.render-scratch/rwstore-*/png``): the scratch's pictures are the renderer's working copies.
        if any(is_scratch_dir(part) for part in relative.parts[:-1]):
            continue
        if not path.is_file():
            continue
        found.append(root / relative)
    return sorted(found, key=lambda path: path.as_posix())


def describe(root: str = "<--out>", *, sep: str | None = None,
             episode: bool = False) -> str:
    """The one sentence a script author needs, for --help and docs.

    ``sep`` defaults to THIS platform's separator, which is what the
    line a render PRINTS needs: that line pastes the reader's own
    ``--out`` -- ``mycase-out\\png\\run-...Z`` on Windows -- in front of
    this template, and a forward-slash template behind a backslash path
    read as two paths glued together (UX finding N24).

    Committed documentation and ``--help`` pass ``sep="/"`` instead,
    because those strings are shared by every reader on every platform:
    a page regenerated on Windows must not ship backslashes to somebody
    running Linux.

    ``episode=True`` is for a render whose inputs come from a nest that
    retires and re-arms, and it is the door's business to know which it
    has: the sentence is printed BEFORE any picture is drawn so a script
    can watch one path, and naming a path one segment short of the one
    the frames arrive at is worse than naming none.  It stays OFF by
    default because that is what every lifecycle-free render writes, and
    the default sentence is the one already published.
    """

    mark = os.sep if sep is None else sep
    if not episode:
        return (f"{root}{mark}<domain>{mark}<product>{mark}<valid-day>{mark}"
                f"<file>.png (domain as d02-3km / d05-111m / native_grid, "
                f"valid-day as YYYY-MM-DD)")
    return (f"{root}{mark}<domain>{mark}<episode>{mark}<product>{mark}"
            f"<valid-day>{mark}<file>.png (domain as d02-3km / d05-111m "
            f"/ native_grid, episode as episode-002, valid-day as "
            f"YYYY-MM-DD)")


__all__ = [
    "DEFAULT_LAYOUT", "DEFAULT_RENDER_PRODUCTS", "EPISODE_PREFIX", "FLAT", "LAYOUTS", "NATIVE_GRID",
    "NESTED", "UNCLASSIFIED", "UNDATED", "HistoryFrame", "deliver",
    "delivered_name", "describe", "engine_name", "engine_output_time",
    "episode_number", "episode_segment", "fs_path", "history_frames",
    "iter_rendered", "parse_engine_output", "place", "probe_delivery_root",
    "product_dir", "valid_day",
]
