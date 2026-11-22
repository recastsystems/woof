"""The two refusals, both directions, plus the product stamp.

A refusal is only as good as the cases it does NOT fire on.  The whole
risk in this program is a gate that trips on an ordinary
``pip install recast-woof`` -- a person whose install is perfectly correct,
told it is broken, with a remedy that does not apply to them.  That is
strictly worse than the defect being guarded, so every test here comes
in a pair: the shape that must refuse, and the shape that must pass in
silence.

Install shapes are built out of REAL files, reusing
``tests/test_provenance``'s builders rather than a second set: a real
package directory, a real ``.dist-info`` read by a real
``PathDistribution``, a real ``git init``.  The renderer bridges are
real files too, carrying real ``GPUWM_BRIDGE_SOURCE_REV=<40 hex>``
byte stamps of the kind ``build.rs`` injects, read back by the same
``woof.bridge_assets.embedded_source_revisions`` the release cut uses.

Each refusal is mutually falsifying by construction: a gate that always
returned ``None`` would fail every ``_refuses`` test, and one that
always refused would fail every ``_is_silent`` test.  Two further tests
hold everything fixed and flip exactly ONE fact -- the bridge's stamp,
and the tree's declared version -- and require the verdict to move,
because "the paths differ" is a weak discriminator and this project has
been burned by instruments that were right for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_provenance import (_dist_info, _git_init, _package, _pip_install,
                             _pyproject)

from woof import provenance_gate
from woof.bridge_assets import SOURCE_REV_MARKER
from woof.provenance import UNKNOWN_VERSION, describe_provenance

ENV = "WOOF_RW_WRFBATCH"

#: Two well-formed 40-hex revisions, distinguishable at a glance.
REV_A = "a" * 40
REV_B = "b" * 40


# ---------------------------------------------------------------------------
# real install shapes
# ---------------------------------------------------------------------------

def _wheel(tmp: Path, *, version="1.8.7"):
    """A plain ``pip install recast-woof``: no source tree, no git, no crate."""

    site = tmp / "site-packages"
    package = _package(site)
    return describe_provenance(package, _dist_info(site, version=version),
                               reported_version=version)


def _checkout(tmp: Path, *, declared="1.8.7", installed=None,
              reported=None, name="woof"):
    """A real git checkout, optionally with an editable install over it.

    ``installed=None`` is the plain clone nobody pip-installed, which is
    where a BORROWED version comes from: ``reported`` is then whatever
    ``woof.__version__`` produced out of a stranger's metadata.
    """

    source = tmp / "home" / "user" / name
    package = _package(source)
    _pyproject(source, version=declared)
    _git_init(source)
    distribution = None
    if installed is not None:
        distribution = _dist_info(tmp / "site-packages", version=installed,
                                  editable_at=source)
    return describe_provenance(package, distribution,
                               reported_version=reported)


def _stamped(path: Path, revision: str | None) -> Path:
    """A real file shaped like a built bridge, with or without a stamp."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"MZ\x90\x00" + b"\x00" * 64
    if revision is not None:
        payload += SOURCE_REV_MARKER + revision.encode("ascii")
    payload += b"\x00" * 64
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# refusal (a): the version contradicts the source it claims to describe
# ---------------------------------------------------------------------------

def test_stale_editable_over_newer_source_refuses(tmp_path):
    """The field report's exact shape: metadata one release, tree another.

    A ``pip install --upgrade`` over an editable install rewrites the
    ``.dist-info`` and leaves the old tree serving every import -- or,
    as here, the tree moves on and the metadata does not.  Two numbers,
    one tree, and until this gate nothing compared them.
    """

    prov = _checkout(tmp_path, declared="1.9.0", installed="1.8.7",
                     reported="1.8.7")
    refusal = provenance_gate.version_identity_refusal(prov)
    assert refusal is not None
    assert "REFUSING" in refusal
    assert "1.9.0" in refusal and "1.8.7" in refusal
    # The remedy has to be actionable and name THIS tree.
    assert "pip install -e" in refusal
    assert str(prov.source_root) in refusal


def test_borrowed_version_that_disagrees_refuses_and_names_the_borrowing(
        tmp_path):
    """No distribution provides this code, yet a number arrived anyway.

    The number came out of some other ``.dist-info``.  When it also
    DIFFERS from what the tree declares, everything this process stamps
    -- the wrfout's GPUWM_VERSION, every receipt, the plot label -- is
    describing a release that is not running.
    """

    prov = _checkout(tmp_path, declared="1.9.0", installed=None,
                     reported="1.6.2")
    refusal = provenance_gate.version_identity_refusal(prov)
    assert refusal is not None
    assert "NO installed distribution provides" in refusal
    assert "1.6.2" in refusal and "1.9.0" in refusal


def test_a_plain_wheel_is_silent(tmp_path):
    """The normal case, and the one a false positive would ruin.

    A wheel ships no pyproject.toml: pip wrote its code and its metadata
    together, so there is exactly one claim and nothing to contradict.
    No git, no crate, no refusal, no output.
    """

    prov = _wheel(tmp_path, version="1.8.7")
    assert provenance_gate.version_identity_refusal(prov) is None
    assert prov.install_kind == "wheel"
    assert prov.git is None


def test_an_uninstalled_clone_is_silent(tmp_path):
    """``git clone`` and run it: ``0+unknown`` is accurate, not a conflict.

    This is the project's own development shape and the shape of anyone
    trying woof without installing it.  Refusing it over "0+unknown is
    not 1.9.0" would refuse a non-defect, loudly, on first contact.
    """

    prov = _checkout(tmp_path, declared="1.9.0", installed=None,
                     reported=UNKNOWN_VERSION)
    assert provenance_gate.version_identity_refusal(prov) is None


def test_a_borrowed_version_that_agrees_is_silent(tmp_path):
    """Borrowed but equal: recorded, never refused.

    A worktree beside an editable install of the same release reports a
    number that came from the wrong metadata and happens to be right.
    It is worth recording -- ``receipt_block`` carries
    ``metadata_is_borrowed`` -- but nothing about the run is wrong, and
    this is the state of every worktree in this repository.
    """

    prov = _checkout(tmp_path, declared="1.8.7", installed=None,
                     reported="1.8.7")
    assert prov.metadata_is_borrowed is True
    assert provenance_gate.version_identity_refusal(prov) is None


def test_a_matched_editable_install_is_silent(tmp_path):
    prov = _checkout(tmp_path, declared="1.8.7", installed="1.8.7",
                     reported="1.8.7")
    assert prov.install_kind == "editable"
    assert provenance_gate.version_identity_refusal(prov) is None


def test_only_the_version_moves_and_the_verdict_moves_with_it(tmp_path):
    """Discrimination: same paths, same tree, one digit different.

    "The two installs are at different paths" would pass the firing test
    above for the wrong reason.  Here the paths are identical in both
    arms and the ONLY difference is the number the metadata carries, so
    a gate that is not actually comparing versions fails.
    """

    agreeing = _checkout(tmp_path / "a", declared="1.8.7",
                         installed="1.8.7", reported="1.8.7")
    conflicting = _checkout(tmp_path / "b", declared="1.8.7",
                            installed="1.6.2", reported="1.6.2")
    assert provenance_gate.version_identity_refusal(agreeing) is None
    assert provenance_gate.version_identity_refusal(conflicting) is not None


def test_the_diagnostic_commands_are_never_refused(tmp_path):
    """A gate that blocks its own diagnostic leaves no way out.

    ``woof version`` exists to report exactly this disagreement and
    ``woof doctor`` reports it as a check; refusing them would leave a
    reader with a broken install and no command that still runs.
    """

    prov = _checkout(tmp_path, declared="1.9.0", installed="1.8.7",
                     reported="1.8.7")
    for command in sorted(provenance_gate.DIAGNOSTIC_COMMANDS):
        provenance_gate.require_consistent_version(command, prov)
    with pytest.raises(ValueError, match="REFUSING"):
        provenance_gate.require_consistent_version("run", prov)


#: The modules ``import woof`` and this gate's verdict run, copied into a
#: throwaway checkout so the verdict is measured on a real interpreter
#: with nothing of this machine's own installs in play.
_GATE_MODULES = ("__init__.py", "_env.py", "cupy_windows_warning.py", "numerics.py", "explain.py",
                 "provenance.py", "provenance_gate.py", "runtime_manifest.py")


@pytest.mark.parametrize("owned", [False, True],
                         ids=["no-install-provides-it", "installed"])
def test_a_stranger_gpuwm_on_the_path_refuses_only_where_nothing_owns_the_code(
        tmp_path, owned):
    """Reading the owning install's version launders no borrowed number.

    A stranger's woof .dist-info stands FIRST on the path.  A checkout no
    distribution provides still wears the stranger's number, because
    ``woof.__version__`` falls back to the first by name, so this gate
    must still call it borrowed and refuse it against the checkout's own
    pyproject.toml.  The same code installed by its own .dist-info reads
    its own version and runs: it used to read the stranger's too, and was
    refused as disagreeing with the distribution that provides it.  A
    real interpreter under ``-S``: only these two directories are on its
    path.
    """

    import os
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[1] / "woof"
    files = {name: (source / name).read_bytes() for name in _GATE_MODULES}
    checkout = tmp_path / "checkout"
    if owned:
        _pip_install(checkout, version="7.7.7", files=files)
    else:
        for name, payload in files.items():
            (checkout / "woof").mkdir(parents=True, exist_ok=True)
            (checkout / "woof" / name).write_bytes(payload)
    _pyproject(checkout, version="7.7.7")
    _pip_install(tmp_path / "user-site", version="0.0.1", files={})
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path / "user-site"), str(checkout)))
    program = (
        "import json, woof;"
        "from woof import provenance_gate as g;"
        "p = g.resolve();"
        "print(json.dumps({'version': woof.__version__,"
        " 'file': woof.__file__, 'borrowed': p.metadata_is_borrowed,"
        " 'refusal': g.version_identity_refusal(p)}))")
    completed = subprocess.run(
        [sys.executable, "-S", "-P", "-c", program], capture_output=True,
        text=True, cwd=str(tmp_path), env=environment, timeout=120)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert Path(payload["file"]).resolve() == (
        checkout / "woof" / "__init__.py").resolve()
    if owned:
        assert payload["version"] == "7.7.7"
        assert payload["borrowed"] is False
        assert payload["refusal"] is None
    else:
        assert payload["version"] == "0.0.1"
        assert payload["borrowed"] is True
        assert payload["refusal"] is not None
        assert "REFUSING" in payload["refusal"]
        assert "'0.0.1'" in payload["refusal"]
        assert "'7.7.7'" in payload["refusal"]


# ---------------------------------------------------------------------------
# refusal (b): a renderer bridge from a tree that is not this one
# ---------------------------------------------------------------------------

def test_a_silently_discovered_foreign_bridge_refuses(tmp_path,
                                                      monkeypatch):
    """#106: the shared staging directory belongs to no tree.

    ``find_renderer`` returns the first candidate that is a FILE, and
    its last candidate is ``~/.woof/bridges``.  A checkout with no
    local build borrows whichever engine another tree staged there, and
    the two do not draw the same catalog.
    """

    monkeypatch.delenv(ENV, raising=False)
    prov = _checkout(tmp_path)
    bridge = _stamped(tmp_path / "shared" / "bridges" / "rw_wrfbatch.exe",
                      REV_B)

    match = provenance_gate.bridge_tree_match(bridge, env_var=ENV, prov=prov)
    assert match.verdict == "foreign"
    assert match.matched is False
    assert match.bridge_source_rev == REV_B

    refusal = provenance_gate.renderer_bridge_refusal(
        bridge, env_var=ENV, prov=prov)
    assert refusal is not None
    assert "REFUSING" in refusal
    assert str(bridge) in refusal
    assert REV_B in refusal
    # Both routes out are named: build it here, or declare that one.
    assert "cargo build" in refusal
    assert f"{ENV}=" in refusal
    with pytest.raises(ValueError, match="REFUSING"):
        provenance_gate.require_matched_renderer(bridge, env_var=ENV,
                                                 prov=prov)


def test_a_bridge_built_from_this_head_passes(tmp_path, monkeypatch):
    """Staged outside the tree, but stamped with this checkout's commit.

    That is what ``woof fetch-bridges`` produces on a checkout sitting
    at the release commit, and it is the same proof the release cut
    itself uses (``bridge_assets.verify_source_revision``).
    """

    monkeypatch.delenv(ENV, raising=False)
    prov = _checkout(tmp_path)
    head = prov.git["commit_full"]
    bridge = _stamped(tmp_path / "shared" / "rw_wrfbatch.exe", head)

    match = provenance_gate.bridge_tree_match(bridge, env_var=ENV, prov=prov)
    assert match.verdict == "stamp-matches"
    assert match.matched is True
    assert provenance_gate.renderer_bridge_refusal(
        bridge, env_var=ENV, prov=prov) is None


def test_only_the_stamp_moves_and_the_verdict_moves_with_it(tmp_path,
                                                            monkeypatch):
    """Discrimination: one bridge path, two stamps, two verdicts.

    Same engine tree, same directory, same filename, same bytes but for
    the forty hex characters.  A gate that is really keying on "the
    binary is outside the source root" would give the same answer twice.
    """

    monkeypatch.delenv(ENV, raising=False)
    prov = _checkout(tmp_path)
    bridge = tmp_path / "shared" / "rw_wrfbatch.exe"

    _stamped(bridge, prov.git["commit_full"])
    assert provenance_gate.bridge_tree_match(
        bridge, env_var=ENV, prov=prov).matched is True

    _stamped(bridge, REV_A)
    assert provenance_gate.bridge_tree_match(
        bridge, env_var=ENV, prov=prov).matched is False


def test_a_bridge_inside_the_executing_tree_passes_without_reading_bytes(
        tmp_path, monkeypatch):
    """``tools/rustwx/target/release`` is this tree's build by definition.

    No stamp is consulted, which is what keeps the everyday development
    loop -- edit, run, the binary is three commits old -- from tripping
    a gate at every invocation.
    """

    monkeypatch.delenv(ENV, raising=False)
    prov = _checkout(tmp_path)
    bridge = _stamped(
        Path(prov.source_root) / "tools" / "rustwx" / "target" / "release"
        / "rw_wrfbatch.exe", REV_A)  # a stamp that does NOT match HEAD

    match = provenance_gate.bridge_tree_match(bridge, env_var=ENV, prov=prov)
    assert match.verdict == "in-tree"
    assert match.matched is True
    assert match.bridge_source_rev is None, (
        "an in-tree binary is matched by location; its bytes are not read")
    assert provenance_gate.renderer_bridge_refusal(
        bridge, env_var=ENV, prov=prov) is None


def test_an_explicitly_declared_bridge_passes(tmp_path, monkeypatch):
    """Naming it in the environment IS the declaration.

    ``rustwx.find_renderer`` already treats this variable as explicit
    configuration -- an override naming a missing file is a hard error
    rather than a fall-through -- so a person who points it at a
    hand-built or hand-copied engine has chosen, and the gate refuses
    only the case where nobody chose.
    """

    prov = _checkout(tmp_path)
    bridge = _stamped(tmp_path / "elsewhere" / "rw_wrfbatch.exe", REV_B)
    monkeypatch.setenv(ENV, str(bridge))

    match = provenance_gate.bridge_tree_match(bridge, env_var=ENV, prov=prov)
    assert match.verdict == "declared"
    assert provenance_gate.renderer_bridge_refusal(
        bridge, env_var=ENV, prov=prov) is None


def test_a_wheel_install_with_a_staged_bridge_is_silent(tmp_path,
                                                        monkeypatch):
    """THE case a false positive would ruin, and it is the common one.

    A pip wheel ships no Rust sources and cannot build any, so the only
    engine it will ever have is one staged into the shared bridge
    directory by ``woof fetch-bridges``.  There is no tree to match it
    against, the question is unanswerable, and an unanswerable question
    must produce silence rather than suspicion.
    """

    monkeypatch.delenv(ENV, raising=False)
    prov = _wheel(tmp_path)
    bridge = _stamped(tmp_path / "shared" / "rw_wrfbatch.exe", REV_B)

    match = provenance_gate.bridge_tree_match(bridge, env_var=ENV, prov=prov)
    assert match.verdict == "unanswerable"
    assert match.matched is True
    assert provenance_gate.renderer_bridge_refusal(
        bridge, env_var=ENV, prov=prov) is None


def test_a_renderer_inside_an_installed_wheel_is_not_called_in_tree(
        tmp_path, monkeypatch):
    """hrrr-full-09 (c): a pip install printed "engine bridge in-tree (the
    renderer is inside the executing source root .../site-packages)".

    A wheel has no source tree; site-packages is where pip put it.  The
    renderer the wheel ships under ``woof/libexec/bridges`` is said to be
    inside the installed wheel's package, and it still passes.
    """

    monkeypatch.delenv(ENV, raising=False)
    prov = _wheel(tmp_path)
    assert prov.install_kind == "wheel"
    bridge = _stamped(Path(prov.package_path) / "libexec" / "bridges"
                      / "rw_wrfbatch", REV_B)

    match = provenance_gate.bridge_tree_match(bridge, env_var=ENV, prov=prov)
    assert match.verdict == "in-package"
    assert match.verdict in provenance_gate.BRIDGE_VERDICTS
    assert match.matched is True
    assert "in-tree" not in match.verdict
    assert "source root" not in match.basis
    assert "installed woof wheel" in match.basis
    assert str(Path(prov.package_path)) in match.basis
    assert provenance_gate.renderer_bridge_refusal(
        bridge, env_var=ENV, prov=prov) is None

    # Elsewhere in site-packages is not the wheel's own package: the
    # question stays unanswerable, never "in-tree".
    beside = _stamped(Path(prov.package_path).parent / "other"
                      / "rw_wrfbatch", REV_B)
    assert provenance_gate.bridge_tree_match(
        beside, env_var=ENV, prov=prov).verdict == "unanswerable"


def test_doctor_names_a_wheel_install_as_a_wheel_install(tmp_path, monkeypatch):
    """The doctor row said "vs this checkout" on a pip install too."""

    from woof import doctor, provenance, rustwx

    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(rustwx.RENDERER_ENV, raising=False)
    prov = _wheel(tmp_path)
    bridge = _stamped(Path(prov.package_path) / "libexec" / "bridges"
                      / "rw_wrfbatch", REV_B)
    monkeypatch.setattr(provenance, "resolve", lambda **_: prov)
    monkeypatch.setattr(rustwx, "find_renderer", lambda: bridge)
    check = doctor._renderer_tree_check()
    assert "vs this wheel install" in check.name
    assert "checkout" not in check.name
    assert check.status == "verified"
    assert check.detail.startswith("in-package: ")


def test_a_source_tree_with_no_git_is_silent(tmp_path, monkeypatch):
    """An unpacked sdist has a crate but no history to compare against."""

    monkeypatch.delenv(ENV, raising=False)
    source = tmp_path / "gpuwm-1.8.7"
    package = _package(source)
    _pyproject(source, version="1.8.7")            # no _git_init here
    prov = describe_provenance(package, None,
                               reported_version=UNKNOWN_VERSION)
    assert prov.git is None
    bridge = _stamped(tmp_path / "shared" / "rw_wrfbatch.exe", REV_B)
    assert provenance_gate.bridge_tree_match(
        bridge, env_var=ENV, prov=prov).verdict == "unanswerable"
    assert provenance_gate.renderer_bridge_refusal(
        bridge, env_var=ENV, prov=prov) is None


def test_an_absent_renderer_is_not_this_gates_business(tmp_path):
    """"Not built" already has its own message, with a build one-liner."""

    prov = _checkout(tmp_path)
    match = provenance_gate.bridge_tree_match(None, env_var=ENV, prov=prov)
    assert match.verdict == "absent"
    assert provenance_gate.renderer_bridge_refusal(
        None, env_var=ENV, prov=prov) is None


def test_an_unstamped_bridge_is_named_as_unstamped(tmp_path, monkeypatch):
    """A binary predating the stamp cannot prove anything about itself.

    It is still refused -- nothing ties it to this tree -- but the
    refusal says WHICH of the two problems it is, because "no stamp" and
    "wrong stamp" have different causes.
    """

    monkeypatch.delenv(ENV, raising=False)
    prov = _checkout(tmp_path)
    bridge = _stamped(tmp_path / "shared" / "rw_wrfbatch.exe", None)
    match = provenance_gate.bridge_tree_match(bridge, env_var=ENV, prov=prov)
    assert match.matched is False
    assert "no GPUWM_BRIDGE_SOURCE_REV" in match.basis


# ---------------------------------------------------------------------------
# the product stamp
# ---------------------------------------------------------------------------

def test_executing_version_prefers_the_code_over_a_borrowed_number(tmp_path):
    """What goes on a product must name what ran, not what pip claims."""

    prov = _checkout(tmp_path, declared="1.9.0", installed=None,
                     reported="1.6.2")
    assert provenance_gate.executing_version(prov) == "1.9.0"


def test_executing_version_of_a_wheel_is_its_metadata(tmp_path):
    prov = _wheel(tmp_path, version="1.8.2")
    assert provenance_gate.executing_version(prov) == "1.8.2"


def test_executing_version_says_it_does_not_know_rather_than_guessing(
        tmp_path):
    """No pyproject, no distribution, a borrowed number: unknown.

    Preferring the borrowed number here is what would put another
    tree's release on this tree's output.
    """

    package = _package(tmp_path / "loose")
    prov = describe_provenance(package, None, reported_version="1.6.2")
    assert prov.metadata_is_borrowed is True
    assert provenance_gate.executing_version(prov) == UNKNOWN_VERSION


def test_the_plot_label_carries_the_executing_version():
    """The only place on a PNG where the producing build can be named.

    Both engines print this string verbatim -- matplotlib through
    ``plot_context``, the rust engine through ``--source-label``, which
    ``rw_wrfbatch`` renders as its ``subtitle_right``.
    """

    from woof.render import DEFAULT_SOURCE_LABEL, default_source_label

    label = default_source_label()
    assert label.startswith(f"{DEFAULT_SOURCE_LABEL} ")
    # (WOOF 1.1.7) the WRF engine's maps name it; the MPAS engine's name theirs the same way
    assert DEFAULT_SOURCE_LABEL == "WOOF-ArWen"
    from woof.hex.render_door import default_source_label as hex_source_label
    assert hex_source_label() == f"WOOF-HEX-A {provenance_gate.executing_version()}"
    assert label.endswith(provenance_gate.executing_version())
    # The ensemble product door builds the same stamp without importing
    # woof.render (the RW-WPS wheel ships rustwx.py, not render.py).
    from woof.rustwx import product_source_label
    assert product_source_label() == label


def test_the_wrfout_producer_stamp_is_the_executing_version():
    """``GPUWM_VERSION`` is the version stamped into every history file."""

    from woof.io.wrfout import _producer_version

    assert _producer_version() == provenance_gate.executing_version()


# ---------------------------------------------------------------------------
# the receipt and the banner
# ---------------------------------------------------------------------------

def test_the_receipt_block_is_json_safe_and_carries_the_verdict(tmp_path):
    """Receipts embed this; a Path in it fails json.dumps months later."""

    prov = _checkout(tmp_path, declared="1.8.7", installed="1.8.7",
                     reported="1.8.7")
    block = provenance_gate.receipt_block(prov)
    text = json.dumps(block)                    # must not raise
    assert json.loads(text) == block
    assert block["schema"] == provenance_gate.RECEIPT_SCHEMA
    assert block["executing_version"] == "1.8.7"
    assert block["consistent"] is True
    assert block["version_identity_refused"] is False
    assert block["install_kind"] == "editable"
    assert block["git"]["commit"]


def test_the_receipt_separates_inconsistent_from_refused(tmp_path):
    """A borrowed-but-agreeing number is recorded, not refused.

    The resolver calls it inconsistent -- the number describes another
    install -- while the gate lets the run proceed, because nothing
    about the run is wrong.  One field cannot say both, and a receipt
    that collapsed them would either hide the loan or invent a refusal.
    This is the live state of every worktree in this repository.
    """

    prov = _checkout(tmp_path, declared="1.8.7", installed=None,
                     reported="1.8.7")
    block = provenance_gate.receipt_block(prov)
    assert block["metadata_is_borrowed"] is True
    assert block["consistent"] is False
    assert block["version_identity_refused"] is False
    assert block["executing_version"] == "1.8.7"


def test_the_banner_prints_once_and_can_be_silenced(monkeypatch, capsys):
    """One line per process, on stderr, naming the command.

    stderr because several of these front doors write JSON or an event
    stream to stdout, where a banner is a parse error.
    """

    monkeypatch.delenv(provenance_gate.BANNER_ENV, raising=False)
    provenance_gate.reset_announcement()
    provenance_gate.announce("woof run")
    provenance_gate.announce("woof run")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("woof run: woof ") == 1

    monkeypatch.setenv(provenance_gate.BANNER_ENV, "0")
    provenance_gate.reset_announcement()
    provenance_gate.announce("woof run")
    assert capsys.readouterr().err == ""
    provenance_gate.reset_announcement()


def test_announce_for_main_returns_the_refusal_instead_of_raising(
        tmp_path, monkeypatch):
    """The standalone console scripts have no refusal boundary.

    ``woof.cli.main`` catches ValueError and prints one sentence at
    exit 2; ``rw-wps`` and the two prepared runners do not, and a
    traceback is not a refusal.
    """

    prov = _checkout(tmp_path, declared="1.9.0", installed="1.8.7",
                     reported="1.8.7")
    monkeypatch.setattr(provenance_gate, "resolve", lambda **_: prov)
    provenance_gate.reset_announcement()
    message = provenance_gate.announce_for_main("rw-wps")
    assert message is not None
    assert "REFUSING" in message
    # Rendered at the terse layer: the explain sentinel never reaches a
    # terminal, and the pointer names the command the reader just ran.
    assert "[[explain]]" not in message
    assert "rw-wps --explain" in message
    provenance_gate.reset_announcement()


# ---------------------------------------------------------------------------
# Companion packages from another release (D4, 2026-10-05)
# ---------------------------------------------------------------------------


def _installed_tilestream(site: Path, version: str) -> Path:
    (site / "tilestream").mkdir(parents=True)
    (site / "tilestream" / "__init__.py").write_text("", encoding="utf-8")
    info = site / f"recast_woof-{version}.dist-info"
    info.mkdir()
    (info / "RECORD").write_text(
        "woof/__init__.py,,\ntilestream/__init__.py,sha256=x,0\n",
        encoding="utf-8")
    return site


def _roots(monkeypatch, gpuwm_root: Path, tilestream_root: Path) -> None:
    roots = {"woof": gpuwm_root, "tilestream": tilestream_root}
    monkeypatch.setattr(provenance_gate, "_companion_root", roots.get)


def test_a_base_venv_tilestream_of_another_release_refuses(tmp_path,
                                                           monkeypatch):
    """The box venv shape: 2.8.6 source woof, 2.8.5 wheel tilestream."""

    prov = _checkout(tmp_path, declared="2.8.6")
    site = _installed_tilestream(tmp_path / "gvenv" / "site-packages",
                                 "2.8.5")
    _roots(monkeypatch, Path(prov.source_root), site)
    refusal = provenance_gate.companion_identity_refusal(prov)
    assert refusal is not None and "REFUSING" in refusal
    assert "2.8.5" in refusal and "2.8.6" in refusal and str(site) in refusal
    with pytest.raises(ValueError, match="mixed releases"):
        provenance_gate.require_consistent_version("run", prov)
    provenance_gate.require_consistent_version("doctor", prov)


def test_a_companion_from_the_same_root_is_silent(tmp_path, monkeypatch):
    prov = _checkout(tmp_path, declared="2.8.6")
    _roots(monkeypatch, Path(prov.source_root), Path(prov.source_root))
    assert provenance_gate.companion_identity_refusal(prov) is None


def test_a_companion_of_the_same_release_elsewhere_is_silent(tmp_path,
                                                            monkeypatch):
    prov = _checkout(tmp_path, declared="2.8.6")
    site = _installed_tilestream(tmp_path / "shared" / "site-packages",
                                 "2.8.6")
    _roots(monkeypatch, Path(prov.source_root), site)
    assert provenance_gate.companion_identity_refusal(prov) is None


def test_a_companion_whose_release_cannot_be_read_is_silent(tmp_path,
                                                           monkeypatch):
    prov = _checkout(tmp_path, declared="2.8.6")
    loose = tmp_path / "loose"
    (loose / "tilestream").mkdir(parents=True)
    _roots(monkeypatch, Path(prov.source_root), loose)
    assert provenance_gate.companion_identity_refusal(prov) is None


def test_the_live_process_has_no_companion_split():
    assert provenance_gate.companion_identity_refusal() is None
