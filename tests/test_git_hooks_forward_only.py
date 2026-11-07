"""The forward-only git hooks in tools/git-hooks refuse amend, rebase,
reset and force push, and leave every forward operation alone.

The project policy, "Forward commits only": no reset, rebase, amend or force-push.
Origin (2026-10-20): two agents chained `git commit --amend` seconds after
a commit.  Each case below is the real git command in a scratch repository
with the hooks installed by tools/git-hooks/install.sh, plus the hook run
directly with the three argument shapes git uses for prepare-commit-msg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "tools" / "git-hooks"
AMEND_MSG = "amend is banned here (forward commits only); make a new commit"


def _sh() -> str:
    sh = shutil.which("sh")
    if sh is None and os.name == "nt":
        exec_path = subprocess.run(
            ["git", "--exec-path"], capture_output=True, text=True, check=True
        ).stdout.strip()
        cand = Path(exec_path).parents[2] / "bin" / "sh.exe"
        sh = str(cand) if cand.exists() else None
    if sh is None:
        pytest.skip("no POSIX sh to run git hooks with")
    return sh


def _env(home: Path) -> dict:
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("GIT_"):
            del env[k]
    env.update(
        HOME=str(home),
        GIT_CONFIG_GLOBAL=str(home / "gitconfig"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@localhost",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@localhost",
    )
    return env


def _git(repo: Path, env: dict, *args: str, ok: bool = True):
    p = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env
    )
    if ok is not None:
        assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout, p.stderr)
    return p


@pytest.fixture()
def repo(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "gitconfig").write_text("[init]\n\tdefaultBranch = main\n")
    env = _env(home)
    r = tmp_path / "r"
    subprocess.run(["git", "init", "-q", str(r)], check=True, env=env)
    p = subprocess.run(
        [_sh(), str(HOOKS / "install.sh"), str(r)],
        capture_output=True, text=True, env=env,
    )
    assert p.returncode == 0, p.stderr
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "base")
    return r, env


def test_commit_amend_is_refused_and_forward_commits_pass(repo):
    r, env = repo
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "x")
    head = _git(r, env, "rev-parse", "HEAD").stdout.strip()
    for amend in (
        ["--amend", "--no-edit"],
        ["--amend", "-m", "y"],
        ["--amen", "-m", "y"],
        ["--amend", "--reset-author", "--no-edit"],
    ):
        p = _git(r, env, "commit", "-q", "--allow-empty", *amend, ok=False)
        assert AMEND_MSG in p.stderr, (amend, p.stderr)
        assert _git(r, env, "rev-parse", "HEAD").stdout.strip() == head
    # Forward commits that look like an amend to the arguments alone.
    _git(r, env, "commit", "-q", "--allow-empty", "-C", "HEAD")
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "--amend is only text")
    _git(r, env, "commit", "-q", "--allow-empty", "-am", "--amend")
    assert _git(r, env, "rev-list", "--count", "HEAD").stdout.strip() == "5"


def test_merge_cherry_pick_revert_squash_pass(repo):
    r, env = repo
    _git(r, env, "switch", "-q", "-c", "side")
    (r / "a.txt").write_text("a\n")
    _git(r, env, "add", "a.txt")
    _git(r, env, "commit", "-q", "-m", "side a")
    side = _git(r, env, "rev-parse", "HEAD").stdout.strip()
    _git(r, env, "switch", "-q", "main")
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "main b")
    _git(r, env, "merge", "-q", "--no-ff", "--no-edit", "side")
    _git(r, env, "revert", "--no-edit", "HEAD", "-m", "1")
    _git(r, env, "cherry-pick", side)
    _git(r, env, "switch", "-q", "-c", "sq", "HEAD~3")
    _git(r, env, "merge", "-q", "--squash", "side")
    _git(r, env, "commit", "-q", "--no-edit")
    _git(r, env, "commit", "-q", "--allow-empty", "-c", side, "--no-edit")


def test_rebase_is_refused(repo):
    r, env = repo
    _git(r, env, "switch", "-q", "-c", "side")
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "side x")
    side = _git(r, env, "rev-parse", "HEAD").stdout.strip()
    _git(r, env, "switch", "-q", "main")
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "main y")
    for cmd in (["rebase", "main", "side"], ["rebase", "-i", "main", "side"]):
        p = _git(r, env, *cmd, ok=False)
        assert "rebase is banned here (forward commits only)" in p.stderr
    assert _git(r, env, "rev-parse", "side").stdout.strip() == side


def test_force_push_is_refused_fast_forward_passes(repo, tmp_path):
    r, env = repo
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, env=env)
    _git(r, env, "remote", "add", "origin", str(bare))
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "x")
    _git(r, env, "push", "-q", "origin", "main")
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "y")
    _git(r, env, "push", "-q", "origin", "main")  # fast-forward
    _git(r, env, "push", "-q", "origin", "HEAD:refs/heads/new")  # new ref
    for force in (["--force", "origin", "HEAD~1:main"], ["origin", "+HEAD~2:main"]):
        p = _git(r, env, "push", "-q", *force, ok=False)
        assert "force push is banned here (forward commits only)" in p.stderr
    remote_main = subprocess.run(
        ["git", "-C", str(bare), "rev-parse", "main"],
        capture_output=True, text=True, env=env,
    ).stdout.strip()
    assert remote_main == _git(r, env, "rev-parse", "HEAD").stdout.strip()
    _git(r, env, "config", "forwardonly.refuseAllPush", "true")
    p = _git(r, env, "push", "-q", "origin", "HEAD:refs/heads/other", ok=False)
    assert "push is banned from this repository" in p.stderr


def test_prepare_commit_msg_argument_shapes(repo):
    """The hook run directly (no git command line above it), with the
    one-, two- and three-argument shapes git uses."""
    r, env = repo
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "x")
    hook = r / ".git" / "hooks" / "prepare-commit-msg"
    msg = r / ".git" / "COMMIT_EDITMSG"
    msg.write_text("m\n")
    prev = _git(r, env, "rev-parse", "HEAD~1").stdout.strip()
    head = _git(r, env, "rev-parse", "HEAD").stdout.strip()

    def run(*args):
        return subprocess.run(
            [_sh(), str(hook), str(msg), *args],
            cwd=r, capture_output=True, text=True, env=env,
        )

    assert run().returncode == 0
    for source in ("message", "template", "merge", "squash"):
        assert run(source).returncode == 0, source
    assert run("commit", prev).returncode == 0  # -c/-C <older commit>
    for named in ("HEAD", head):  # --amend
        p = run("commit", named)
        assert p.returncode == 1 and AMEND_MSG in p.stderr, named


def test_existing_hook_is_chained_not_lost(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = _env(home)
    r = tmp_path / "r"
    subprocess.run(["git", "init", "-q", str(r)], check=True, env=env)
    own = r / ".git" / "hooks" / "prepare-commit-msg"
    own.write_text('#!/bin/sh\necho chained >> "$1"\n', newline="\n")
    own.chmod(0o755)
    for _ in range(2):  # re-running refreshes, keeps the chain
        p = subprocess.run(
            [_sh(), str(HOOKS / "install.sh"), str(r)],
            capture_output=True, text=True, env=env,
        )
        assert p.returncode == 0, p.stderr
    assert (r / ".git" / "hooks" / "prepare-commit-msg.local").exists()
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "a")
    assert "chained" in _git(r, env, "log", "-1", "--format=%B").stdout


RESET_MSG = "reset is banned here (forward commits only)"


def test_reset_and_backward_branch_moves_are_refused(repo):
    r, env = repo
    for i in range(3):
        _git(r, env, "commit", "-q", "--allow-empty", "-m", f"c{i}")
    head = _git(r, env, "rev-parse", "HEAD").stdout.strip()
    _git(r, env, "branch", "keep")
    for cmd in (
        ["reset", "-q", "--hard", "HEAD~1"],
        ["reset", "-q", "--mixed", "HEAD~1"],
        ["reset", "-q", "--soft", "HEAD~2"],
        ["branch", "-f", "keep", "HEAD~1"],
        ["update-ref", "refs/heads/main", "HEAD~1"],
    ):
        p = _git(r, env, *cmd, ok=False)
        assert RESET_MSG in p.stderr, (cmd, p.stderr)
        for b in ("main", "keep"):
            assert _git(r, env, "rev-parse", b).stdout.strip() == head, cmd
    # Sideways (another line, not an ancestor either way) is refused too.
    _git(r, env, "switch", "-q", "-c", "other", "HEAD~2")
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "other")
    _git(r, env, "switch", "-q", "main")
    p = _git(r, env, "reset", "-q", "--hard", "other", ok=False)
    assert "sideways" in p.stderr
    p = _git(r, env, "checkout", "-q", "-B", "main", "HEAD~1", ok=False)
    assert RESET_MSG in p.stderr
    assert _git(r, env, "rev-parse", "main").stdout.strip() == head


def test_forward_ref_moves_pass(repo):
    r, env = repo
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "a")
    _git(r, env, "branch", "behind", "HEAD~1")             # new branch
    _git(r, env, "switch", "-q", "-c", "side")
    (r / "s.txt").write_text("s\n")
    _git(r, env, "add", "s.txt")
    _git(r, env, "commit", "-q", "-m", "side")              # commit
    _git(r, env, "switch", "-q", "main")                    # checkout
    _git(r, env, "commit", "-q", "--allow-empty", "-m", "b")
    _git(r, env, "merge", "-q", "--no-ff", "--no-edit", "side")  # merge
    _git(r, env, "switch", "-q", "behind")
    _git(r, env, "merge", "-q", "--ff-only", "main")         # fast-forward
    _git(r, env, "branch", "-f", "side", "main")             # forward -f
    _git(r, env, "checkout", "-q", "--detach", "HEAD~2")     # detached HEAD
    _git(r, env, "reset", "-q", "--hard", "HEAD~1")          # detached reset
    _git(r, env, "switch", "-q", "main")
    _git(r, env, "reset", "-q", "--hard", "HEAD")            # no move
    (r / "s.txt").write_text("changed\n")
    _git(r, env, "stash", "-q")                              # stash
    _git(r, env, "stash", "pop", "-q")
    _git(r, env, "checkout", "-q", "--", "s.txt")
    _git(r, env, "branch", "-m", "behind", "renamed")        # rename
    _git(r, env, "branch", "-D", "renamed")                  # deletion


def test_refused_reset_leaves_the_checkout_as_it_was(repo):
    """reset --hard/--mixed/--keep and checkout -B rewrite the index and
    working tree before they move the branch; a refused move must not
    leave them rewound under an unchanged HEAD."""
    r, env = repo
    f = r / "f.txt"
    for text in ("one\n", "two\n"):
        f.write_text(text)
        _git(r, env, "add", "f.txt")
        _git(r, env, "commit", "-q", "-m", text.strip())
    (r / "g.txt").write_text("g\n")
    _git(r, env, "add", "g.txt")
    _git(r, env, "commit", "-q", "-m", "g")
    head = _git(r, env, "rev-parse", "HEAD").stdout.strip()
    for cmd in (
        ["reset", "-q", "--hard", "HEAD~2"],
        ["reset", "-q", "--mixed", "HEAD~2"],
        ["reset", "-q", "--keep", "HEAD~2"],
        ["checkout", "-q", "-B", "main", "HEAD~2"],
    ):
        p = _git(r, env, *cmd, ok=None)
        assert p.returncode != 0 and RESET_MSG in p.stderr, (cmd, p.stderr)
        assert _git(r, env, "rev-parse", "HEAD").stdout.strip() == head, cmd
        assert _git(r, env, "status", "--porcelain").stdout == "", cmd
        assert f.read_text() == "two\n" and (r / "g.txt").exists(), cmd
    # An uncommitted change survives a refused --mixed and --keep.
    f.write_text("local\n")
    for cmd in (["reset", "-q", "--mixed", "HEAD~2"], ["reset", "-q", "--keep", "HEAD~1"]):
        p = _git(r, env, *cmd, ok=None)
        assert p.returncode != 0, cmd
        assert f.read_text() == "local\n", cmd
        assert _git(r, env, "status", "--porcelain").stdout.split() == ["M", "f.txt"], cmd
