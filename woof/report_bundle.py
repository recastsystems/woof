"""One command that turns "it broke" into a file somebody can read.

``woof report`` collects, from a run directory, the artifacts that
actually explain a failure -- the receipts, the failure capsule, the
worker stderr, the stage logs, the printed command chain if one was
recorded, this install's provenance, the runtime versions, the card, and
the free space on the volumes involved -- redacts every identity-shaped
string out of them, and writes a plain zip the reporter can open before
they send it.

Three properties are the whole point, and each exists because the
obvious implementation gets it wrong.

**It survives the failure it exists for.**  The defect this was written
against is out-of-space: the decoder routes every diagnostic through two
error-ignoring writes to the SAME filesystem that just failed, and the
consumer's unwind removes the staging directory holding the gate note,
so the reporter is left with ``RuntimeError: pipeline producer exited
1:`` -- an empty message, no log, no argv.  A collector that streams a
staging tree onto that same full volume, or that raises the first time
an artifact it expected is missing, is useless exactly there.  So the
archive is built in memory and written once, relocating to another
volume if the first write is refused; a missing artifact is a named line
in the manifest, never an exception; and when a volume is nearly full
the manifest says so next to the note that the empty producer message is
a KNOWN erasure rather than additional information.  Nothing here fixes
that defect -- that is a separate lane -- but nothing here pretends the
bundle is complete when it is not.

**It is anonymous, and says how.**  Usernames, home directories,
hostnames, IP and MAC addresses, e-mail addresses, credential-shaped
strings, and every environment variable outside an explicit allowlist
are replaced by class placeholders -- inside JSON receipts and log text
as well as in path names.  Redaction runs in two passes: the first
HARVESTS identity literals, including any account name found as a
home-directory path segment anywhere in the collected bytes, which is
how a foreign username planted in a config value teaches the redactor to
scrub the same name out of a log line where it appears bare.  The second
pass applies them.

**A chosen domain is science, not identity.**  Latitudes, longitudes,
dates, grid shapes, physics choices and file names are what the reporter
asked the model to compute; they are kept verbatim, and the manifest
says so, so nobody has to guess whether their study area was in the
file.

The routes do not all leave the same artifacts (the front door writes a
supervisor progress file and a failure capsule; the prepared runners
write ``report.json``; the tree runner writes ``evidence/``), so absence
is reported per artifact WITH the route that would have written it.  A
bundle that is thin because the run died in its first minute has to look
different from one that is thin because the collector gave up.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import datetime as _datetime
import errno
from fnmatch import fnmatch
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile

from woof import DISTRIBUTION_NAME, __version__
from woof.explain import layered, warn

#: The document this command writes.  Versioned because somebody reading
#: a bundle six months from now has to know what shape it is.
BUNDLE_SCHEMA = "gpuwm-diagnostic-report-v1"

#: Bytes kept from the head and the tail of a text file.  A failure lives
#: at the END of a log and the invocation lives at the START, so a single
#: window would lose one of them; the middle is elided with a marker that
#: says how much went.
_TEXT_HEAD_BYTES = 8 * 1024
_TEXT_TAIL_BYTES = 128 * 1024

#: Receipts are kept whole up to here.  Set from a measured artifact,
#: not from a guess: a real prepared single-domain ``report.json`` for a
#: 6 h HRRR forecast is 320 KiB, and a receipt truncated at 256 KiB is a
#: JSON document that no longer parses -- which costs the typed failure
#: the collector came for.  A receipt that still exceeds this is stored
#: head-only and SAID to be unparseable, rather than looking whole.
_JSON_BYTES = 1024 * 1024

#: The longest environment value kept whole.  ``NVIDIA_REQUIRE_CUDA`` is
#: 2.5 KiB of driver-brand permutations on a stock CUDA image and turned
#: the manifest's environment section into a wall.
_ENV_VALUE_BYTES = 240

#: The whole archive is assembled in memory so no scratch file is needed
#: on a volume that may be the one that just filled up.  This is the cap
#: that keeps that promise affordable.
_TOTAL_BYTES = 8 * 1024 * 1024

#: Directory-walk bounds.  A run directory can hold thousands of frames;
#: the inventory is a list of names and sizes, not a copy.
_MAX_DEPTH = 5
_MAX_INVENTORY = 4000

# ---------------------------------------------------------------------------
# The deny-set: refused before anything opens, reads, hashes or lists it
#
# Owner ruling, and the reason this is a separate mechanism from
# redaction rather than a corner of it: "a secret that was read and then
# scrubbed is still a secret we read."  Redaction is the SECOND line of
# defence.  The first is that a credential file is structurally
# uncollectable -- so these names are refused ahead of every syscall, and
# a refusal is counted by class rather than named, because a file NAME
# can itself be the secret.
# ---------------------------------------------------------------------------

#: Directories never descended into.  Every dot-directory is refused as
#: well; these are named because they are the ones whose contents are
#: unambiguously somebody's credentials, and because naming them makes
#: the rule readable rather than merely effective.
_DENY_DIRS = frozenset({
    ".aws", ".ssh", ".docker", ".azure", ".kube", ".gcloud", ".gnupg",
    ".config", ".npm", ".cargo", ".rustup", ".kaggle", ".pki", ".gem",
    ".m2", ".terraform", ".local", ".netrc.d", "secrets", "secret",
    "credentials", "private", "keys", ".git", ".hg", ".svn", ".tox",
    "__pycache__", "node_modules", "site-packages", "dist-packages",
    "venv", ".venv", "virtualenv", ".idea", ".vscode", ".cache",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
})

#: Exact file names refused outright.
_DENY_NAMES = frozenset({
    ".netrc", "_netrc", ".git-credentials", ".npmrc", ".pypirc",
    ".dockercfg", ".boto", ".s3cfg", ".htpasswd", ".pgpass", ".my.cnf",
    "credentials", "credential", "creds", "token", "tokens", "secret",
    "secrets", "password", "passwords", "kubeconfig", "authorized_keys",
    "known_hosts", "service-account.json", "id_rsa", "id_dsa",
    "id_ecdsa", "id_ed25519", "identity",
})

#: Name SHAPES refused outright.  Matched case-insensitively against the
#: file name, so ``prod-credentials.json`` and ``api.key`` are refused on
#: the strength of what they say they are.
_DENY_PATTERNS = (
    "*.pem", "*.key", "*.p12", "*.pfx", "*.keytab", "*.jks", "*.ppk",
    "*.asc", "*.gpg", "*.kdbx", "*.crt", "*.cer", "*.der", "*.pkcs12",
    "*_rsa", "*_dsa", "*_ecdsa", "*_ed25519", "*.env", ".env.*",
    "*credential*", "*secret*", "*token*", "*password*", "*passwd*",
    "*apikey*", "*api_key*", "*.htpasswd", "*keystore*", "*truststore*",
)

#: Reasons a path is refused, as the manifest reports them.  Counted by
#: class; never accompanied by the name that earned the count.
_REFUSAL_PRIVATE_DIR = "inside a private-configuration directory"
_REFUSAL_DOT = "a dot-file or dot-directory, which no woof route writes"
_REFUSAL_NAME = "a credential file name"
_REFUSAL_SHAPE = "a credential-shaped file name"
_REFUSAL_ESCAPE = "resolves outside the run directory (symlink or ..)"
_REFUSAL_UNRESOLVABLE = "could not be resolved, so it was not opened"

#: What the product itself writes, and therefore the ONLY things whose
#: CONTENT is read.  Collection is by allowlist, not by sweep: anything
#: not named here is inventoried by name and size at most.  Patterns are
#: matched against the path RELATIVE to the run directory, so
#: ``evidence/run-receipt.json`` is named as such rather than admitting
#: every ``*.json`` at any depth.
_COLLECT = (
    ("receipt", (
        "report.json", "progress.json", "run-progress.json",
        "failure-capsule.json", "certification-capsule.json",
        "evidence/*.json", "*/report.json", "*/progress.json",
        "*receipt*.json", "*/*receipt*.json",
        "*target.json", "*/*target.json",
    )),
    ("config", (
        "*.toml", "namelist.input", "namelist.wps", "*/namelist.input",
        "*/namelist.wps", "*.namelist",
    )),
    ("log", (
        "*.log", "*/*.log", "*.err", "*/*.err", "*.out", "*/*.out",
        "gate.txt", "*/gate.txt", "failure.ready", "*/failure.ready",
    )),
    ("command", (
        "*next-step*", "*command*", "*.sh", "*.ps1", "*.cmd",
    )),
)

#: Evidence that a directory is a woof run directory at all.  The gate
#: asks for a product ARTIFACT rather than specifically a receipt: the
#: out-of-space defect leaves a zero-byte decoder log and a staging gate
#: note and no receipt at all, and refusing to collect that would close
#: the exact case this command exists for.
#: Deliberately NOT ``*.log``, ``*.err`` or ``*.toml``.  A generic log or
#: a TOML file is not this product's signature -- a source checkout has a
#: pyproject.toml and a home directory collects stray logs, and admitting
#: either would hand back most of the class this gate exists to close.
#: Every entry below is a name a woof route actually writes.  A reporter
#: whose output went to a file of their own naming is one flag away:
#: ``--log`` is them naming their evidence explicitly, and it bypasses
#: this gate because an explicit path is not a sweep.
_RUN_DIR_MARKERS = (
    "report.json", "progress.json", "run-progress.json",
    "failure-capsule.json", "certification-capsule.json",
    "evidence/*.json", "worker-*.log", "worker-*/*.log",
    "gate.txt", "*/gate.txt", "failure.ready", "*/failure.ready",
    "decoder*.log", "*/decoder*.log", "pipeline*.log", "pipeline*.err",
    "wrfout*", "wrfout/*", "gpuwmrst_*", "*/gpuwmrst_*",
    "namelist.input", "namelist.wps", "*target.json",
    "prepared-cache/*", "native/*",
)

#: Free space below which out-of-space is the first suspect.
_TIGHT_DISK_BYTES = 64 * 1024 * 1024

#: Distinct warning lines printed in the manifest.  Anything past this is
#: COUNTED and said so, never dropped in silence: the whole reason the
#: warnings are here is that a run continuing past something is what
#: explains where it ended up, and a truncation nobody is told about
#: hides exactly that.
_MAX_WARNINGS = 40

#: Environment variables whose VALUES are collected.  Everything else is
#: reported as a name only, and only when the name itself is clean.
_ENV_PREFIXES = ("WOOF_", "GPUWM_", "CUPY_", "CUDA_", "NVIDIA_", "NCCL_", "RW_WPS_")
_ENV_NAMES = frozenset({
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED", "PYTHONPATH", "PYTHONWARNINGS", "VIRTUAL_ENV",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "LANG", "LC_ALL", "TZ",
    "SHELL", "COMSPEC", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
})

#: Account names that identify a role rather than a person.  Registering
#: one as a literal would rewrite ordinary prose -- "root domain" is not
#: somebody's home directory -- and none of them narrows a bundle to a
#: person.  Their HOME PATHS are still collapsed by the path rules above,
#: so the superuser's tree reads as ``<home>/...`` whether or not the
#: account name is also a literal.
_GENERIC_ACCOUNTS = frozenset({
    "root", "admin", "administrator", "user", "users", "home", "public",
    "default", "defaultuser", "ubuntu", "debian", "centos", "ec2-user",
    "docker", "runner", "jenkins", "build", "test", "guest", "system",
    "localhost", "all users", "public user", "opt", "usr", "var", "tmp",
})

#: The route label for an artifact every completed forecast writes,
#: whichever runner wrote it.  Its absence is a gap under any route.
_EVERY_ROUTE = "every completed forecast route"

#: What a woof run leaves behind: ``(glob, route, what it would tell us)``.
#: Grouped by route because the routes differ -- the front door writes a
#: supervisor progress file and, on a crash, a failure capsule; the
#: prepared single-domain runner writes ``report.json``; the tree runner
#: writes into ``evidence/``.  Reporting a tree artifact as "missing"
#: after a front-door run would be noise, so the manifest names the route
#: beside every absence and states which route it thinks ran.
_EXPECTED = (
    ("report.json", "prepared single-domain forecast",
     "the run receipt: inputs, physics, output inventory, authority "
     "digests -- and on a failed run its error_type/error/traceback"),
    ("progress.json", "prepared single-domain forecast",
     "the status file: PASS/FAIL/RUNNING, frames written, elapsed"),
    ("evidence/run-receipt.json", "prepared domain-tree forecast",
     "the tree run receipt: per-domain authorities and digests"),
    ("evidence/failed-run-receipt.json", "prepared domain-tree forecast",
     "the tree failure receipt: error_type, error, traceback"),
    ("evidence/progress.json", "prepared domain-tree forecast",
     "the tree status file"),
    # The route is the one the Missing section matches against the
    # detected route, which is the front door whenever a failure capsule
    # is found; ``woof go`` writes the same file for its forecast
    # stage, and that belongs in the description, not in the route.
    ("run-progress.json", "woof run (supervised front door)",
     "the heartbeat woof run writes, and woof go writes for its "
     "forecast stage: phase, elapsed, last durable wrfout, "
     "last checkpoint"),
    ("failure-capsule.json", "woof run (supervised front door)",
     "the supervisor's failure record: exception type, message, "
     "traceback, last phase and step, GPU identity"),
    ("worker-*.stderr.log", "woof run (supervised front door)",
     "the worker's captured stderr -- with its stdout pair, the only "
     "log files this product writes to disk"),
    ("worker-*.stdout.log", "woof run (supervised front door)",
     "the worker's captured stdout"),
    ("certification-capsule.json", _EVERY_ROUTE,
     "the certification capsule: route, run shape, output digests"),
)

#: The receipt that identifies which route was taken, most specific first.
_ROUTE_MARKERS = (
    ("evidence/run-receipt.json", "prepared domain-tree forecast"),
    ("evidence/failed-run-receipt.json", "prepared domain-tree forecast"),
    ("report.json", "prepared single-domain forecast"),
    ("evidence/progress.json", "prepared domain-tree forecast"),
    ("progress.json", "prepared single-domain forecast"),
    ("failure-capsule.json", "woof run (supervised front door)"),
    ("run-progress.json", "woof run (supervised front door)"),
)

#: Receipt keys that carry a failure, in the order a route writes them.
_FAILURE_KEYS = (("error_type", "error", "traceback"),
                 ("exception_type", "exception_message", "traceback"),
                 ("error_type", "error_message", "traceback"))

#: The empty-message signature the out-of-space erasure leaves behind.
_ERASED_PRODUCER = re.compile(
    r"pipeline producer exited\s+\d+\s*:\s*$", re.MULTILINE)

_TRACEBACK_HEAD = "Traceback (most recent call last):"

#: A line this product printed in its own voice.  MOST woof failures
#: are not tracebacks: the CLI's refusal boundary turns every documented
#: ValueError into ``woof <command>: <message>`` and exits 2 with no
#: traceback at all, on purpose.  A collector that only recognised
#: ``Traceback (most recent call last):`` therefore reported "no typed
#: failure" for the single most common way a run ends -- observed live,
#: against a real refusal on node 15, which is why this exists.
_PRODUCT_MESSAGE = re.compile(
    r"^((?:woof|rw-wps)(?:[- ][a-z0-9-]+)*):\s+(\S.*)$", re.MULTILINE)

#: Product messages that report success.  The refusal shape and the
#: receipt shape are the same sentence, so the ones that say a thing
#: WORKED must not be offered as the reason a run failed.
_NOT_A_REFUSAL = re.compile(
    r"(?i)^(pass\b|ok\b|wrote\b|writing\b|done\b|continuing\b|"
    r"skipped\b|runtime estate\b)")

#: One warning line, as :func:`woof.explain.warn` prints it.  Warn-not-
#: block means a run can pass THROUGH something worth knowing about, so
#: the warnings a run printed are part of what led up to a failure.
_WARNING = re.compile(r"^warning:\s+(\S.*)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_WINDOWS_HOME = re.compile(
    r"(?i)([A-Za-z]:(?:\\{1,4}|/)+Users(?:\\{1,4}|/)+)"
    r"([^\\/\s\"'<>|:*?,;\]\)]+)")
_POSIX_HOME = re.compile(
    r"(?<![\w.])/(?:home|Users)/([^/\s\"'<>|:*?,;\]\)]+)")
_ROOT_HOME = re.compile(r"(?<![\w.])/root(?![A-Za-z0-9_])")

#: The same Windows profile as it is reached from a Linux box that has
#: the drive mounted.  Neither rule above sees it: the drive-anchored
#: one wants a drive letter and a colon, and the POSIX one refuses a
#: ``/Users`` segment that has a word character in front of it -- which
#: ``/mnt/c`` is.  A record written from such a box therefore kept the
#: account name under a rule set that says it removes home-directory
#: prefixes of any absolute path.
#:
#: Replacement only, never harvested from: this is the forward-slash
#: spelling, and learning an account name from a forward-slash ``Users``
#: segment is the REST-path mistake :data:`_ANY_USERS_SEGMENT` exists to
#: avoid.  The prefix carries the name away with it in any case.
_WSL_HOME = re.compile(
    r"(?i)/mnt/[a-z]/users/([^/\s\"'<>|:*?,;\]\)]+)")

#: HARVEST ONLY, and deliberately broader than the collapse rules above.
#: A run directory nested under a second ``Users`` segment -- which is
#: what a temporary directory and plenty of real trees look like --
#: carries an account name the drive-anchored rule cannot see.  Learning
#: it here and letting the LITERAL rule replace it keeps the readable
#: shape (``...Users\<user>\run``) instead of collapsing a whole prefix
#: that was not actually a home directory.
#:
#: BACKSLASHES ONLY, which is the whole reason this is a separate
#: pattern from :data:`_POSIX_HOME`.  The forward-slash form also
#: matches a REST path: a URL with a ``Users`` collection segment would
#: teach the redactor that the next segment -- often a plain word like
#: "data" -- is an account name, and every later mention of that word in
#: every collected log would be rewritten to ``<user>``.  The POSIX rule
#: already refuses that case through its own lookbehind, so nothing is
#: lost by declining to look at forward slashes here.
_ANY_USERS_SEGMENT = re.compile(
    r"(?i)\\{1,4}Users\\{1,4}([^\\/\s\"'<>|:*?,;\]\)]+)")

_MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
_IPV4 = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b")
# Six to eight hextet groups, or a ``::`` compressed form that has at
# least one hextet on one side of it.  Two deliberate narrowings, both
# from real text this collector reads:
#
#   * NOT three-to-five groups -- ``12:00:00`` is a clock, not an
#     address, and a collector that mangles every timestamp is worse
#     than one that misses a link-local literal.
#   * NOT a bare ``::`` -- the classifier line
#     ``"License :: OSI Approved :: Apache Software License"`` is in
#     this project's own pyproject.toml, and a rule that ate it would
#     rewrite every table and log that uses `` :: `` as a separator.
_IPV6 = re.compile(
    r"\b(?:[0-9A-Fa-f]{1,4}:){5,7}[0-9A-Fa-f]{1,4}\b"
    r"|(?<![\w:])[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*::"
    r"(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*)?(?![\w:])"
    r"|(?<![\w:])::[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*(?![\w:])")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_PRIVATE_HOST = re.compile(
    r"(?i)\b[a-z0-9][a-z0-9-]{1,62}\.(?:local|lan|internal|intranet|"
    r"corp|home|localdomain)\b")

_PEM = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
    r"-----END [A-Z ]*PRIVATE KEY-----")
# The leading boundary is a lookbehind on [A-Za-z0-9], NOT ``\b``.  ``_``
# is a word character, so ``\b`` fails on exactly the spelling these keys
# have in practice: ``db_password=``, ``my_api_key:``, ``GPUWM_SECRET=``,
# ``aws_secret_access_key=``.  Every underscore-prefixed credential in
# every log this collector reads went through that hole.
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(api[_-]?key|apikey|secret[_-]?key|secret|"
    r"token|password|passwd|pwd|access[_-]?key(?:[_-]?id)?|"
    r"private[_-]?key|client[_-]?secret|auth[_-]?token|authorization|"
    r"session[_-]?id|credentials?|creds)(?![A-Za-z0-9])"
    r"(\s*[:=]\s*\\?\"?'?)([^\s\"',;}\]]+)")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{8,}=*")
_TOKEN_SHAPES = re.compile(
    r"\b(?:"
    r"[sprk]k[-_](?:live|test|prod)[-_]?[A-Za-z0-9_-]{6,}"
    r"|[sprk]k[-_][A-Za-z0-9]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|A(?:KIA|SIA|IDA|ROA)[0-9A-Z]{12,}"
    r"|AIza[A-Za-z0-9_\-]{20,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}"
    r"|\d{4,8}:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}"
    r")")
# The last-resort rule: a long ALPHANUMERIC run that mixes case and
# digits.  Two exclusions carry the weight.
#
# Hex is safe by construction.  A lowercase SHA-256 has no uppercase and
# an uppercase one has no lowercase, so neither matches -- digests are
# the identity this project reports with and they must survive.
#
# ``_`` and ``-`` are NOT in the class, and that is the fix for a real
# defect rather than a nicety: with them included this rule ate this
# project's own scientific vocabulary. ``RW_volumetric_soil_moisture_0``,
# ``module_mp_morr_two_moment_O2vec`` and a physics-profile token were
# all replaced by ``<redacted-credential>`` in a bundle that promised to
# keep physics choices verbatim -- and each one then counted as a
# credential the reporter never had.  Separators split those names into
# sub-threshold pieces; real opaque keys have no separators to split on.
_HIGH_ENTROPY = re.compile(
    r"(?<![A-Za-z0-9_\-])(?=[A-Za-z0-9]{28,}(?![A-Za-z0-9]))"
    r"(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*\d)"
    r"[A-Za-z0-9]{28,}(?![A-Za-z0-9_\-])")

#: An environment variable NAME that says its value is a secret.  Applied
#: to allowlisted names only -- everything outside the allowlist already
#: loses its value -- and it drops the value outright rather than trying
#: to recognise it, because a credential's value is by design
#: indistinguishable from noise.
_SECRET_NAME = re.compile(
    r"(?i)(secret|token|password|passwd|pwd|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|creds|auth|signature|session)")

#: What each class is replaced by.  Readable words in angle brackets, so
#: a reader can see WHAT was removed where instead of finding a hole.
PLACEHOLDERS = {
    "home_directory": "<home>",
    "username": "<user>",
    "hostname": "<host>",
    "ipv4_address": "<ip>",
    "ipv6_address": "<ipv6>",
    "mac_address": "<mac>",
    "email_address": "<email>",
    "credential": "<redacted-credential>",
}

#: Prose for the manifest: what each class means, in the reporter's terms.
CLASS_DESCRIPTIONS = {
    "home_directory": "home-directory prefixes of any absolute path",
    "username": "account names (yours, and any other found in a path)",
    "hostname": "machine names, FQDNs and private-domain names",
    "ipv4_address": "IPv4 addresses",
    "ipv6_address": "IPv6 addresses",
    "mac_address": "MAC addresses",
    "email_address": "e-mail addresses",
    "credential": "API keys, tokens, passwords, private-key blocks",
    "environment_variable": "environment variables outside the allowlist "
                            "(names kept when clean, values never)",
}

#: Said in the manifest, because a reporter deserves to know what a
#: bundle CANNOT settle before they wait on an answer from it.
LIMITS = (
    "It carries no model output.  wrfout frames, restart files and NPZ "
    "caches are listed by name and size only, so nothing here can settle "
    "a question about the numbers in a field.",
    "It carries no input data.  GRIB, met_em and static tiles are named "
    "by their receipts' digests, never copied.",
    "A run that died before writing its first receipt leaves nothing but "
    "an inventory and this machine section -- the manifest says so "
    "rather than looking complete.",
    "An unsupervised run (`woof run --no-supervise`) writes no log file "
    "at all; its stderr exists only in the terminal it ran in.  Re-run "
    "under the supervisor, or pass the redirected file with --log.",
    "Redaction is deliberate loss.  If your report depends on which "
    "machine or which account, that is exactly what was removed.",
    "Anonymity is by class, not by review.  Your own account, home "
    "directory and machine name come from what this box reports about "
    "itself; a SECOND person's name is removed once it appears as a path "
    "segment anywhere in what was collected, and a name that appears "
    "only as a bare word in prose, in nothing that looks like a path, "
    "cannot be told from any other word.  This is why the bundle is a "
    "readable zip: read it before you send it.",
)


#: The four shapes a home-directory prefix takes, in the order
#: :func:`redact_home_directories` applies them.  Named once so that the
#: next shape is added in one place and every caller gets it.
HOME_DIRECTORY_RULES = (_WSL_HOME, _WINDOWS_HOME, _POSIX_HOME, _ROOT_HOME)


def redact_home_directories(text, *, on_hit=None):
    """``text`` with every home-directory prefix replaced by ``<home>``.

    The one home-path rule in the tree.  :meth:`Redactor.apply` runs it
    as part of a whole identity sweep, and the release gates
    (``tools/release/precut_gate.py``) run it alone over the records
    they write, so that a receipt cannot carry the scratch directory,
    the interpreter path or the traceback of the box the reading was
    taken on.  A second copy of these patterns would be a second rule
    to keep in step, which is how a shape gets fixed in one place and
    ships from the other.

    ``on_hit`` is called once per replacement, for a caller that counts
    what it removed.
    """

    if not text:
        return text

    def _sub(_match):
        if on_hit is not None:
            on_hit()
        return PLACEHOLDERS["home_directory"]

    for pattern in HOME_DIRECTORY_RULES:
        text = pattern.sub(_sub, text)
    return text


def _identity_seeds(environ) -> list[tuple[str, str]]:
    """Literal identity strings this machine can name about itself."""

    seeds: list[tuple[str, str]] = []
    for key in ("USER", "USERNAME", "LOGNAME", "USERDOMAIN"):
        value = environ.get(key)
        if value:
            seeds.append((value, "username"))
    for key in ("HOSTNAME", "COMPUTERNAME", "HOST"):
        value = environ.get(key)
        if value:
            seeds.append((value, "hostname"))
    try:
        import getpass

        seeds.append((getpass.getuser(), "username"))
    except Exception:  # noqa: BLE001 - identity is best effort, never fatal
        pass
    for probe in (socket.gethostname, socket.getfqdn):
        try:
            name = probe()
        except Exception:  # noqa: BLE001
            continue
        if name:
            seeds.append((name, "hostname"))
            head = name.split(".")[0]
            if head != name:
                seeds.append((head, "hostname"))
    try:
        home = Path.home()
    except Exception:  # noqa: BLE001
        home = None
    if home is not None:
        seeds.append((str(home), "home_directory"))
        if home.name:
            seeds.append((home.name, "username"))
    return seeds


class Redactor:
    """Two passes: learn what identifies this box, then remove it.

    :meth:`harvest` is called over every byte the collector intends to
    ship BEFORE :meth:`apply` touches any of it.  That ordering is what
    makes a username planted in one surface -- a config value, a receipt
    path, a log line -- removable from all the others, including the
    surfaces where it appears bare with no path around it to recognise
    it by.
    """

    def __init__(self, *, environ=None) -> None:
        self.counts: Counter[str] = Counter()
        self._literals: dict[str, str] = {}
        self._compiled: list[tuple[re.Pattern[str], str]] | None = None
        environ = os.environ if environ is None else environ
        for value, klass in _identity_seeds(environ):
            self.register(value, klass)

    # -- learning -----------------------------------------------------

    def register(self, value: str, klass: str) -> None:
        """Record one literal to scrub, unless it names a role."""

        value = str(value or "").strip()
        if len(value) < 3:
            return
        if value.lower() in _GENERIC_ACCOUNTS:
            return
        self._literals.setdefault(value, klass)
        self._compiled = None

    def harvest(self, text: str) -> None:
        """Learn the account name any home-shaped path in ``text`` carries."""

        if not text:
            return
        for match in _ANY_USERS_SEGMENT.finditer(text):
            self.register(match.group(1), "username")
        for match in _WINDOWS_HOME.finditer(text):
            self.register(match.group(2), "username")
        for match in _POSIX_HOME.finditer(text):
            self.register(match.group(1), "username")

    # -- applying -----------------------------------------------------

    def _rules(self) -> list[tuple[re.Pattern[str], str]]:
        if self._compiled is not None:
            return self._compiled
        rules: list[tuple[re.Pattern[str], str]] = [
            (_EMAIL, "email_address"),
            (_MAC, "mac_address"),
            # IPv4 BEFORE IPv6: an IPv4-mapped address ``::ffff:a.b.c.d``
            # otherwise has ``::ffff:a`` eaten as IPv6 and the remaining
            # ``.b.c.d`` is no longer four octets, so three of them ship.
            (_IPV4, "ipv4_address"),
            (_IPV6, "ipv6_address"),
        ]
        # Longest first: ``<box>.<domain>`` must go before ``<box>``, and
        # a full home path before the account name inside it.
        #
        # The boundaries exclude only [A-Za-z0-9], NOT ``_``: an account
        # name is routinely glued to a word by an underscore
        # (``<name>_scratch``, ``<NAME>_HOME`` as an environment variable
        # name), and a ``\w`` boundary left every one of those intact.
        for literal in sorted(self._literals, key=len, reverse=True):
            rules.append((
                re.compile(r"(?<![A-Za-z0-9])" + re.escape(literal)
                           + r"(?![A-Za-z0-9])", re.IGNORECASE),
                self._literals[literal]))
        # LAST, so a machine that knows its own FQDN replaces the whole
        # name in one piece rather than having its domain half taken by
        # the generic private-suffix rule and its host half by a literal.
        rules.append((_PRIVATE_HOST, "hostname"))
        self._compiled = rules
        return rules

    def apply(self, text: str) -> str:
        """``text`` with every identity class replaced by its placeholder."""

        if not text:
            return text

        def _counted(klass):
            def _sub(_match):
                self.counts[klass] += 1
                return PLACEHOLDERS[klass]
            return _sub

        # A credential ASSIGNMENT keeps its key, so a reader can see WHICH
        # secret was present; only the value goes.
        def _assignment(match):
            self.counts["credential"] += 1
            return (match.group(1) + match.group(2)
                    + PLACEHOLDERS["credential"])

        # SHAPED secrets before KEYED ones, and this order is load
        # bearing.  ``Authorization: Bearer <token>`` matches the
        # assignment rule with `authorization` as the key and the literal
        # word ``Bearer`` as the value -- so running the assignment rule
        # first replaced the word "Bearer", leaving the token itself
        # intact AND removing the anchor `_BEARER` needed to find it.
        # The token shipped whole.  Recognising the shape first means the
        # secret is gone before anything can eat its label.
        text = _PEM.sub(_counted("credential"), text)
        text = _TOKEN_SHAPES.sub(_counted("credential"), text)
        text = _BEARER.sub(_counted("credential"), text)
        text = _CREDENTIAL_ASSIGNMENT.sub(_assignment, text)
        # Home paths before bare literals: one ``<home>`` is more
        # informative than a drive-anchored profile prefix with only the
        # account name taken out, and it leaves nothing behind to match.
        def _home_hit():
            self.counts["home_directory"] += 1

        text = redact_home_directories(text, on_hit=_home_hit)
        for pattern, klass in self._rules():
            text = pattern.sub(_counted(klass), text)
        return _HIGH_ENTROPY.sub(_counted("credential"), text)

    def path(self, value) -> str:
        """A path for the manifest: separators normalised, identity gone."""

        return self.apply(str(value).replace("\\", "/"))

    def arcname(self, value) -> str:
        """A redacted name that is legal as a file name on Windows.

        The angle brackets that make ``<user>`` readable inside a
        document are among the characters Windows forbids in a file
        name.  Python's own ``zipfile`` sanitises them on extract, but
        Explorer's built-in extractor does not -- and "open it before
        you send it" is the whole promise of shipping a plain zip, on
        the platform most of this product's users are on.
        """

        return (self.apply(str(value).replace("\\", "/"))
                .replace("<", "_").replace(">", "_"))

    def json(self, payload):
        """``payload`` with every string VALUE redacted, keys untouched.

        Structural rather than a dump-redact-reparse round trip: the
        round trip is one regex away from producing a document that no
        longer parses, and the one thing this collector must never do is
        raise while assembling a bundle about somebody else's exception.

        Keys are deliberately left alone.  This is only ever called on
        the manifest's own machine/volume/provenance blocks, whose keys
        are constants in this file -- while an account named ``gpu``,
        ``python``, ``packages`` or ``machine`` is entirely ordinary on a
        compute node, and redacting keys renamed ``machine`` to
        ``<user>`` and killed the whole command with ``KeyError`` in the
        renderer.  Receipts and logs, which DO carry user data in their
        keys, are redacted as text and never come through here.
        """

        if isinstance(payload, dict):
            return {key: self.json(value) for key, value in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [self.json(item) for item in payload]
        if isinstance(payload, str):
            return self.apply(payload)
        if payload is None or isinstance(payload, (bool, int, float)):
            return payload
        return self.apply(str(payload))


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    """One artifact the bundle carries, or declines to carry whole."""

    arcname: str
    category: str
    source: str
    stored_bytes: int
    original_bytes: int | None
    note: str


@dataclass
class Report:
    """A built bundle: its manifest, its prose, and its archive bytes."""

    manifest: dict
    text: str
    archive: bytes
    entries: list[Entry] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    written_path: Path | None = None
    fallback_reason: str | None = None


def _now_stamp(now=None) -> str:
    moment = now or _datetime.datetime.now(_datetime.timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _disk(path) -> dict:
    try:
        usage = shutil.disk_usage(str(path))
    except OSError as error:
        return {"free_bytes": None, "total_bytes": None,
                "error": f"{type(error).__name__}: {error.strerror or error}"}
    return {"free_bytes": int(usage.free), "total_bytes": int(usage.total)}


def _read_window(path: Path, limit_head: int, limit_tail: int):
    """``(text, original_bytes, note)`` for one text file.

    Never raises for a file that vanished, is unreadable, or is not text:
    a collector that dies on the artifact it came for is the defect it
    was written against.
    """

    try:
        size = path.stat().st_size
    except OSError as error:
        return None, None, f"unreadable ({type(error).__name__})"
    try:
        with open(path, "rb") as handle:
            if size <= limit_head + limit_tail:
                raw = handle.read()
                elided = 0
            else:
                head = handle.read(limit_head)
                elided = size - limit_head - limit_tail
                if limit_tail:
                    handle.seek(-limit_tail, os.SEEK_END)
                    tail = handle.read(limit_tail)
                else:
                    tail = b""
                raw = (head
                       + f"\n\n... [{elided} bytes elided by woof report] "
                         f"...\n\n".encode()
                       + tail)
    except OSError as error:
        return None, size, f"unreadable ({type(error).__name__})"
    text = raw.decode("utf-8", errors="replace")
    if not elided:
        note = "whole"
    elif limit_tail:
        note = f"head+tail, {elided} bytes elided"
    else:
        note = (f"head only, {elided} bytes elided -- this JSON is "
                "truncated and will not parse")
    return text, size, note


def _denied_name(name: str) -> str | None:
    """Why this file name is refused, or None.  No filesystem access."""

    lowered = name.lower()
    if lowered.startswith("."):
        return _REFUSAL_DOT
    if lowered in _DENY_NAMES:
        return _REFUSAL_NAME
    if any(fnmatch(lowered, pattern) for pattern in _DENY_PATTERNS):
        return _REFUSAL_SHAPE
    return None


def _denied_directory(name: str) -> str | None:
    """Why this directory is never descended into, or None."""

    lowered = name.lower()
    if lowered.startswith("."):
        return _REFUSAL_DOT
    if lowered in _DENY_DIRS:
        return _REFUSAL_PRIVATE_DIR
    return None


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _refusal(path: Path, root: Path) -> str | None:
    """Why ``path`` is refused before anything opens it, or None.

    RESOLVE FIRST, THEN TEST.  A symlink named ``worker-02.stderr.log``
    pointing at somebody's private key would otherwise pass every
    name-based rule there is: the name the collector wants is not a
    passport, and the only accurate question is what the path actually
    IS.  Both the literal path and the resolved one are tested, so
    neither a friendly name nor a hostile target can carry the other
    past the check.
    """

    try:
        resolved = path.resolve()
    except OSError:
        return _REFUSAL_UNRESOLVABLE
    if not _within(resolved, root):
        return _REFUSAL_ESCAPE
    for candidate in (path, resolved):
        for part in candidate.parts[:-1]:
            reason = _denied_directory(part)
            if reason is not None:
                return reason
        reason = _denied_name(candidate.name)
        if reason is not None:
            return reason
    return None


def _categorise(relative: str) -> str | None:
    """The category whose allowlist claims this relative path, or None.

    ``None`` means the file is not something this product writes, so its
    CONTENT is never read -- it is inventoried by name and size and
    nothing more.
    """

    lowered = relative.lower()
    name = lowered.rsplit("/", 1)[-1]
    for category, patterns in _COLLECT:
        for pattern in patterns:
            if fnmatch(lowered, pattern) or fnmatch(name, pattern):
                return category
    return None


def _walk(root: Path) -> tuple[list[Path], "Counter[str]", bool]:
    """``(files, refused_by_reason, truncated)`` under ``root``.

    Every path is put through :func:`_refusal` BEFORE it is stat-ed,
    opened or recorded, and a refused path is counted by reason and then
    forgotten -- its name never reaches the manifest, because a file
    name can itself be the secret.  Denied directories are not
    descended into at all, so nothing inside them is even enumerated.

    This is the first line of defence, and it is deliberately not the
    redactor.  ``RUNDIR`` defaults to the current directory, so a
    reporter can aim the command one directory too high; scrubbing
    ``.aws/credentials`` after reading it would still mean we read it.
    """

    files: list[Path] = []
    refused: Counter[str] = Counter()
    truncated = False
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if is_dir:
                reason = (_denied_directory(child.name)
                          or _refusal(child, root))
                if reason is not None:
                    refused[reason] += 1
                    continue
                if depth < _MAX_DEPTH:
                    stack.append((child, depth + 1))
                continue
            # The refusal runs BEFORE the file is stat-ed, let alone
            # opened.  A denied path is counted and forgotten; nothing
            # downstream ever learns its name.
            reason = _refusal(child, root)
            if reason is not None:
                refused[reason] += 1
                continue
            if len(files) >= _MAX_INVENTORY:
                truncated = True
                continue
            files.append(child)
    return files, refused, truncated


def _python_environment() -> dict:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in ("numpy", "cupy-cuda12x", "cupy", "netCDF4", "matplotlib",
                 "jsonschema", "wrf-rust", "rasterio", "pyproj",
                 DISTRIBUTION_NAME):
        try:
            versions[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 - absence is the answer
            versions[name] = None
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_build": " ".join(sys.version.replace("\n", " ").split()[1:]),
        "executable": sys.executable,
        "packages": versions,
    }


def _nvidia_driver_version() -> str | None:
    """The kernel-driver version a user would read off ``nvidia-smi``."""

    try:
        completed = subprocess.run(
            [*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    first = completed.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _gpu_environment() -> dict:
    """Card model, memory, driver and CUDA runtime -- never fatally.

    CuPy first because it is the stack the model actually runs on, then
    ``nvidia-smi`` for a base install with no GPU extra.  Neither is
    allowed to raise and neither allocates on the device: a diagnostic
    that needs a working GPU is no use to somebody whose GPU is the
    problem, and a collector must not take memory from whatever else is
    sharing the card.
    """

    from woof.gpu_smi import is_amd, inventory
    if is_amd():
        block = {"probe": "rocm-smi", "backend": "hip", "devices": [], "note": None}
        try:
            block["devices"] = [{"index": int(row["index"]), "name": row["name"],
                "total_memory_bytes": int(row["memory.total"])*1048576,
                "driver_version": row["driver_version"], "pci_bus_id": row["pci.bus_id"]}
                for row in inventory()]
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            block["note"] = f"ROCm inventory unavailable: {type(error).__name__}: {error}"
        return block
    block: dict = {"probe": None, "devices": [], "note": None}
    try:
        import cupy  # noqa: PLC0415 - optional at report time by design
    except Exception as error:  # noqa: BLE001
        block["note"] = (f"CuPy not importable ({type(error).__name__}); "
                         "asked nvidia-smi instead")
    else:
        try:
            count = int(cupy.cuda.runtime.getDeviceCount())
            block["driver_version"] = int(cupy.cuda.runtime.driverGetVersion())
            block["runtime_version"] = int(
                cupy.cuda.runtime.runtimeGetVersion())
            for index in range(count):
                properties = cupy.cuda.runtime.getDeviceProperties(index)
                name = properties.get("name", b"")
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                block["devices"].append({
                    "index": index,
                    "name": str(name),
                    "total_memory_bytes": int(
                        properties.get("totalGlobalMem", 0)),
                    "compute_capability":
                        f"{properties.get('major')}.{properties.get('minor')}",
                })
            block["probe"] = "cupy"
            # CuPy's driverGetVersion is the CUDA driver API version
            # (12080), which is NOT the number a user reads off
            # nvidia-smi or a release note (580.65.06).  Half the
            # "which driver are you on" exchanges are about that
            # difference, so report both and label them.
            block["nvidia_driver_version"] = _nvidia_driver_version()
            return block
        except Exception as error:  # noqa: BLE001
            block["note"] = ("CuPy is present but the device probe failed: "
                             f"{type(error).__name__}: {error}")
    try:
        completed = subprocess.run(
            [*__import__('woof.gpu_smi', fromlist=['command']).command(),
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        block["note"] = (f"{block['note'] or ''}; nvidia-smi unavailable "
                         f"({type(error).__name__})").lstrip("; ")
        return block
    if completed.returncode != 0:
        block["note"] = (f"{block['note'] or ''}; nvidia-smi exited "
                         f"{completed.returncode}").lstrip("; ")
        return block
    block["probe"] = "nvidia-smi"
    for index, line in enumerate(completed.stdout.strip().splitlines()):
        fields = [item.strip() for item in line.split(",")]
        record: dict = {"index": index, "name": fields[0] if fields else None}
        for position, key in ((1, "total_memory"), (2, "driver_version"),
                              (3, "compute_capability")):
            if len(fields) > position:
                record[key] = fields[position]
        block["devices"].append(record)
    return block


def _provenance_block() -> dict:
    """This install's identity, through the product's own resolvers.

    Two answers, because they are two different questions and the
    reader of this bundle is someone trying to reproduce a stranger's
    failure.  ``runtime_manifest.provenance`` is the receipt-binding
    identity ladder (sealed manifest, then checkout, then wheel
    RECORD) and stays exactly where it was.  ``running_tree`` is
    :mod:`woof.provenance_gate`'s answer to "which tree served the
    imports": install kind, branch, sha, dirt, and whether the version
    this install reports agrees with the version its own source
    declares.

    That last field is the one this bundle existed without.  A reporter
    can send a perfect diagnostic zip whose ``gpuwm_version`` names a
    release they never ran, and nothing in the bundle would say so.
    """

    from woof import runtime_manifest

    root = Path(runtime_manifest.__file__).resolve().parent.parent
    block: dict = {"gpuwm_version": __version__, "package_root": str(root)}
    try:
        block.update(runtime_manifest.provenance(root))
    except Exception as error:  # noqa: BLE001 - a broken install still reports
        block["identity_source"] = None
        block["identity_error"] = f"{type(error).__name__}: {error}"
    try:
        from woof.provenance_gate import receipt_block

        block["running_tree"] = receipt_block()
    except Exception as error:  # noqa: BLE001 - a broken install still reports
        block["running_tree"] = {
            "unavailable": f"{type(error).__name__}: {error}"}
    return block


def _environment_variables(environ) -> tuple[dict, list[str]]:
    """``(collected, omitted_names)`` under the allowlist.

    The allowlist is by PREFIX, so it admits names nobody enumerated:
    ``GPUWM_*`` and ``RW_WPS_*`` are exactly where this project's own
    credentials live, and a bare value has no ``key=`` around it for the
    credential rule to recognise once it is a standalone string.  A
    secret-shaped NAME therefore loses its value here, before any
    redaction pass runs -- the allowlist decides what is looked at, and
    this decides what is worth looking at at all.
    """

    collected: dict[str, str] = {}
    omitted: list[str] = []
    for name in sorted(environ):
        if not (name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES):
            omitted.append(name)
            continue
        if _SECRET_NAME.search(name):
            collected[name] = PLACEHOLDERS["credential"]
            continue
        value = str(environ[name])
        if len(value) > _ENV_VALUE_BYTES:
            value = (value[:_ENV_VALUE_BYTES]
                     + f" ... [{len(value) - _ENV_VALUE_BYTES} more "
                       "characters elided]")
        collected[name] = value
    return collected, omitted


def _failure_from_receipt(name: str, payload) -> dict | None:
    """The typed failure a receipt carries, if it carries one."""

    if not isinstance(payload, dict):
        return None
    for kind_key, message_key, trace_key in _FAILURE_KEYS:
        kind = payload.get(kind_key)
        if not isinstance(kind, str) or not kind:
            continue
        message = payload.get(message_key)
        trace = payload.get(trace_key)
        return {
            "source": name,
            "found_in": "receipt",
            "exception_type": kind,
            "message": "" if message is None else str(message),
            "traceback": None if trace is None else str(trace),
        }
    return None


def _refusal_from_log(name: str, text: str) -> dict | None:
    """The last thing the product said in its own voice, if it was bad news.

    Reported with a ``found_in`` that does not overstate it: this is the
    last product message, which for a refusal IS the failure and for a
    run that ended some other way is merely the last thing it said.  The
    reader gets the sentence and the exit code and can tell.
    """

    candidate = None
    for match in _PRODUCT_MESSAGE.finditer(text):
        message = match.group(2).strip()
        if _NOT_A_REFUSAL.match(message):
            continue
        candidate = (match.group(1), message)
    if candidate is None:
        return None
    return {
        "source": name,
        "found_in": "last product message",
        "exception_type": candidate[0],
        "message": candidate[1],
        "traceback": None,
    }


def _failure_from_log(name: str, text: str) -> dict | None:
    """The LAST Python traceback in one log, structured.

    The last one wins: a run that retried has its final failure at the
    end, and that is the one that ended the run.
    """

    index = text.rfind(_TRACEBACK_HEAD)
    if index < 0:
        return None
    block = text[index:]
    terminal = None
    for line in block.splitlines()[1:]:
        if line and not line[0].isspace():
            terminal = line
    if terminal is None:
        return None
    kind, _, message = terminal.partition(":")
    return {
        "source": name,
        "found_in": "log",
        "exception_type": kind.strip(),
        "message": message.strip(),
        "traceback": "\n".join(block.splitlines()[:200]),
    }


def _bytes_human(value) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    if amount < 1024:
        return f"{int(amount)} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        amount /= 1024.0
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
    return f"{amount:.1f} TiB"


def _home_or_none():
    try:
        return Path.home()
    except Exception:  # noqa: BLE001
        return None


def _cwd_or_none():
    """The working directory, or None when it no longer exists.

    ``Path.cwd()`` raises ``FileNotFoundError`` when the directory the
    process started in has been removed underneath it -- which is
    EXACTLY the shape of the defect this command exists for: the
    consumer's unwind removes the staging directory, and a reporter who
    then runs ``woof report`` from a shell still sitting in it would
    have been met with a raw traceback from the collector instead of a
    bundle.
    """

    try:
        return Path.cwd()
    except OSError:
        return None


def _has_any_receipt(directory: Path) -> bool:
    return any((directory / name).exists()
               for name, _ in _ROUTE_MARKERS)


def _resolve_run_dir(run_dir: Path) -> tuple[Path | None, str]:
    """The directory to read, and a sentence when it is not the one asked for.

    Run with no arguments the command reads the current directory.  When
    that directory holds no receipt but the documented default output
    directory below it does, the run below is what the reporter meant --
    and the manifest says which one was read, so nobody has to wonder
    which of two directories the bundle describes.
    """

    if not run_dir.is_dir():
        return None, ""
    # Resolved, so a bundle collected from inside a run with no arguments
    # names the directory it describes instead of reporting ".".  The
    # absolute path is redacted like any other, so this costs no identity.
    try:
        run_dir = run_dir.resolve()
    except OSError:
        pass
    if _has_any_receipt(run_dir):
        return run_dir, ""
    for candidate in (run_dir / "out" / "run", run_dir / "out",
                      run_dir / "run"):
        if candidate.is_dir() and _has_any_receipt(candidate):
            relative = candidate.relative_to(run_dir).as_posix()
            return candidate, (
                f"read {relative}/ below the directory given: it holds the "
                "run receipts and the directory given did not")
    return run_dir, ""


def _demand_run_directory(resolved: Path, found: list[Path],
                          redactor: "Redactor") -> None:
    """Refuse a directory that is not a run, before collecting anything.

    This one rule removes an entire class rather than mitigating it.  A
    home directory, a source checkout or a downloads folder holds
    nothing this product wrote -- so there is nothing here to collect,
    and taking "whatever happens to be there" is how a diagnostic
    collector turns into an exfiltration tool by accident.  The refusal
    names what it looked for, so the reader can see it was looked at.
    """

    relatives = [item.relative_to(resolved).as_posix().lower()
                 for item in found]
    if any(fnmatch(name, marker.lower())
           for name in relatives for marker in _RUN_DIR_MARKERS):
        return
    raise ValueError(layered(
        f"{redactor.path(resolved)} does not look like a woof run "
        "directory, so nothing in it was read.\n"
        "  It looked for: " + ", ".join(_RUN_DIR_MARKERS[:9]) + ", ...\n"
        "  remedy: point it at the directory your run was writing to --\n"
        "    woof report /path/to/run\n"
        "  # or, if the output went somewhere else, name it:\n"
        "    woof report /path/to/run --log /path/to/output.log",
        "A diagnostic collector that gathers whatever it finds is an "
        "exfiltration tool by accident: pointed one directory too high "
        "it would read private configuration, credentials and personal "
        "files that no redaction rule is written to recognise.  So the "
        "collector reads only what this product writes, and only "
        "somewhere this product wrote something.  Refusing here is what "
        "makes the rest of the guarantee checkable."))


def build_report(run_dir, *, exit_code=None, extra_logs=(), environ=None,
                 now=None, redactor=None,
                 require_run_directory: bool = True) -> Report:
    """Collect, redact and assemble -- without writing anything.

    Separated from the write so ``--dry-run`` runs the ENTIRE collection
    and shows the reporter the real manifest rather than a prediction of
    one, and so a refused write can be retried onto another volume
    without re-reading a filesystem that may itself be failing.
    """

    environ = dict(os.environ if environ is None else environ)
    stamp = _now_stamp(now)
    run_dir = Path(run_dir)
    resolved, resolution_note = _resolve_run_dir(run_dir)
    redactor = redactor or Redactor(environ=environ)

    raw: list[dict] = []
    inventory: list[dict] = []
    notes: list[str] = []
    if resolution_note:
        notes.append(resolution_note)

    refused_before_reading: Counter[str] = Counter()
    inventory_truncated = False
    if resolved is None:
        notes.append(
            f"no run directory was read: {run_dir} is not a directory, so "
            "this bundle carries the install and machine sections only")
    else:
        found, refused_before_reading, inventory_truncated = _walk(resolved)
        if require_run_directory and not extra_logs:
            _demand_run_directory(resolved, found, redactor)
        for path in found:
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            relative = path.relative_to(resolved).as_posix()
            inventory.append({"path": relative, "bytes": size})
            category = _categorise(relative)
            if category is None:
                continue
            head, tail = ((_JSON_BYTES, 0) if category == "receipt"
                          else (_TEXT_HEAD_BYTES, _TEXT_TAIL_BYTES))
            text, original, note = _read_window(path, head, tail)
            if text is None:
                notes.append(f"{relative}: {note}")
                continue
            raw.append({"arcname": f"run/{relative}", "relative": relative,
                        "category": category, "source": path, "text": text,
                        "original": original, "note": note})

    for extra in extra_logs or ():
        extra = Path(extra)
        text, original, note = _read_window(
            extra, _TEXT_HEAD_BYTES, _TEXT_TAIL_BYTES)
        if text is None:
            notes.append(f"--log {extra.name}: {note}")
            continue
        raw.append({"arcname": f"logs/{extra.name}", "relative": extra.name,
                    "category": "log", "source": extra, "text": text,
                    "original": original, "note": note})

    # "Missing" means NOT ON DISK, and it is answered from the inventory
    # rather than from what was successfully read.  A receipt that exists
    # but could not be opened was previously reported as absent, so the
    # manifest said "no receipt of any route was found here" four
    # sections above a note saying that receipt was right there and
    # unreadable.  A bundle that contradicts itself is worse than a thin
    # one.  Extra --log files are excluded from both answers: a log
    # handed in from elsewhere must not be able to fake a route.
    on_disk = [item["path"] for item in inventory]
    missing = [
        {"artifact": pattern, "route": artifact_route,
         "would_have_shown": description}
        for pattern, artifact_route, description in _EXPECTED
        if not any(fnmatch(name, pattern) for name in on_disk)]
    route = next((label for marker, label in _ROUTE_MARKERS
                  if any(fnmatch(name, marker) for name in on_disk)), None)

    machine = {
        "os_family": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "python": _python_environment(),
        "gpu": _gpu_environment(),
    }
    collected_env, omitted_env = _environment_variables(environ)
    volumes = {}
    for label, target in (("run_directory", resolved),
                          ("working_directory", _cwd_or_none()),
                          ("temporary_directory", Path(tempfile.gettempdir())),
                          ("home_directory", _home_or_none())):
        if target is None:
            continue
        volumes[label] = {"path": str(target).replace("\\", "/"),
                          **_disk(target)}
    provenance = _provenance_block()

    # ---- pass one: learn what identifies this box -------------------
    for source in (
            *(item["text"] for item in raw),
            *(str(item["source"]) for item in raw),
            *(str(item["source"]).replace("\\", "/") for item in raw),
            *collected_env.values(),
            # NAMES too: an account name shows up as a variable name
            # (`<NAME>_SCRATCH`) at least as often as inside a value, and
            # the omitted list ships names even though it never ships
            # their values.
            *collected_env, *omitted_env,
            *(item["path"] for item in inventory),
            json.dumps(provenance, default=str),
            json.dumps(machine, default=str),
            json.dumps(volumes, default=str),
            str(run_dir), str(resolved or ""), *notes):
        redactor.harvest(source)

    # ---- pass two: apply it -----------------------------------------
    entries: list[Entry] = []
    payloads: list[tuple[str, str]] = []
    failure = None
    log_failure = None
    refusal = None
    passed = False
    warnings: list[str] = []
    erased: list[str] = []
    budget = _TOTAL_BYTES
    used_names: set[str] = set()
    for item in raw:
        safe_name = redactor.arcname(item["arcname"])
        # Two files whose names redact identically -- `alice-notes.log`
        # and `bob-notes.log` both become `_user_-notes.log` -- would
        # otherwise be written twice into the archive under one name.
        # Most extractors keep whichever came last, so the bundle would
        # be silently short one file while the manifest listed both.
        if safe_name in used_names:
            stem, dot, suffix = safe_name.rpartition(".")
            if not dot:
                stem, suffix = safe_name, ""
            counter = 2
            while safe_name in used_names:
                safe_name = f"{stem}-{counter}{dot}{suffix}"
                counter += 1
        used_names.add(safe_name)
        safe_text = redactor.apply(item["text"])
        encoded = safe_text.encode("utf-8")
        if len(encoded) > budget:
            entries.append(Entry(
                safe_name, item["category"], redactor.path(item["source"]),
                0, item["original"],
                "omitted: the bundle size budget was already spent"))
            continue
        budget -= len(encoded)
        payloads.append((safe_name, safe_text))
        entries.append(Entry(safe_name, item["category"],
                             redactor.path(item["source"]), len(encoded),
                             item["original"], item["note"]))
        if item["category"] == "receipt":
            try:
                parsed = json.loads(safe_text)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("status") == "PASS":
                passed = True
            failure = failure or _failure_from_receipt(safe_name, parsed)
        else:
            log_failure = (_failure_from_log(safe_name, safe_text)
                           or log_failure)
            refusal = _refusal_from_log(safe_name, safe_text) or refusal
            for match in _WARNING.finditer(safe_text):
                line = match.group(1).strip()
                if line not in warnings:
                    warnings.append(line)
            if _ERASED_PRODUCER.search(safe_text):
                erased.append(safe_name)
    # A receipt's typed error beats a traceback (it is the run's own
    # verdict), and a traceback beats a refusal line (it carries where).
    #
    # A run whose receipt says PASS, or whose reporter passed
    # --exit-code 0, keeps its refusal-shaped lines OUT of the failure
    # slot.  "woof check: the optional [products] table is absent" is
    # printed in the same voice as a refusal, and offering it as the
    # reason a successful run failed is the collector inventing a
    # problem for somebody who did not have one.
    succeeded = passed or exit_code == 0
    failure = failure or log_failure or (None if succeeded else refusal)

    safe_omitted = [name for name in (redactor.apply(item)
                                      for item in omitted_env)
                    if "<" not in name]
    redactor.counts["environment_variable"] += len(omitted_env)

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": stamp,
        "gpuwm_version": __version__,
        "run_directory": redactor.path(
            resolved if resolved is not None else run_dir),
        "route_detected": route,
        "observed_exit_code": exit_code,
        "included": [
            {"path": entry.arcname, "category": entry.category,
             "source": entry.source, "stored_bytes": entry.stored_bytes,
             "original_bytes": entry.original_bytes, "note": entry.note}
            for entry in entries],
        "missing_expected_artifacts": missing,
        # `path`, not `apply`: a note quotes a path, and one rendered
        # with native separators beside a manifest that normalises them
        # reads as two different machines.
        "notes": [redactor.path(note) for note in notes],
        "inventory": [{"path": redactor.apply(item["path"]),
                       "bytes": item["bytes"]} for item in inventory],
        "inventory_truncated": inventory_truncated,
        # Counted by class, never named.  A file name can itself be the
        # secret -- `prod-stripe-live-key.pem` says everything -- so a
        # manifest that listed what it refused would leak precisely what
        # refusing it was meant to protect.
        "refused_before_reading": dict(refused_before_reading),
        "failure": failure,
        "warnings": warnings[:_MAX_WARNINGS],
        "warnings_withheld": max(0, len(warnings) - _MAX_WARNINGS),
        "erased_producer_message_in": erased,
        "provenance": redactor.json(provenance),
        "machine": redactor.json(machine),
        "volumes": redactor.json(volumes),
        "environment": {
            "collected": {name: redactor.apply(value)
                          for name, value in collected_env.items()},
            "omitted_names": safe_omitted,
            "omitted_count": len(omitted_env),
        },
        "tight_volumes": [label for label, block in volumes.items()
                          if isinstance(block.get("free_bytes"), int)
                          and block["free_bytes"] < _TIGHT_DISK_BYTES],
        "redaction": {
            "classes": {name: int(redactor.counts.get(name, 0))
                        for name in sorted(CLASS_DESCRIPTIONS)},
            "descriptions": dict(CLASS_DESCRIPTIONS),
            "kept_on_purpose":
                "Latitudes, longitudes, projections, dates, grid shapes, "
                "physics choices and file names are kept VERBATIM.  They "
                "describe what you asked the model to compute: scientific "
                "content, not identity.",
        },
        "limits": list(LIMITS),
    }
    text = _render_manifest(manifest)
    archive = _archive_bytes(stamp, text, manifest, payloads)
    return Report(manifest=manifest, text=text, archive=archive,
                  entries=entries, missing=missing)


# ---------------------------------------------------------------------------
# The manifest a human reads before deciding to send the file
# ---------------------------------------------------------------------------

def _wrap(text: str, indent: str = "  ") -> list[str]:
    """One paragraph, wrapped to a terminal that might be 80 columns.

    The manifest is read in a terminal by somebody who is already having
    a bad day; a 400-character line that wraps at an arbitrary column is
    a wall, and this project's whole message convention is against walls.
    """

    import textwrap

    return textwrap.wrap(" ".join(str(text).split()), width=78,
                         initial_indent=indent, subsequent_indent=indent
                         ) or [indent.rstrip()]


def _render_manifest(manifest: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# woof diagnostic report ({manifest['schema']})")
    add("")
    add(f"  created (UTC)   {manifest['created_utc']}")
    add(f"  woof version   {manifest['gpuwm_version']}")
    add(f"  run directory   {manifest['run_directory']}")
    add(f"  route detected  {manifest['route_detected'] or 'none'}")
    provenance = manifest["provenance"]
    add(f"  install         {provenance.get('identity_source')}")
    wheel = provenance.get("installed_wheel") or {}
    if wheel:
        add(f"                  {wheel.get('distribution_name')} "
            f"{wheel.get('distribution_version')}, RECORD "
            f"{str(wheel.get('record_aggregate_sha256'))[:16]}... over "
            f"{wheel.get('record_file_count')} files")
    editable = provenance.get("installed_editable") or {}
    if editable:
        git = editable.get("git") or {}
        add(f"                  {editable.get('distribution_name')} "
            f"{editable.get('distribution_version')}, editable source at "
            f"{editable.get('source_root')}")
        add(f"                  tree is "
            f"{'DIRTY' if git.get('dirty') else 'clean'} "
            f"({git.get('dirty_files')} tracked file(s) modified, "
            f"{git.get('untracked_files')} untracked)")
    if provenance.get("git_commit"):
        add(f"                  commit {provenance['git_commit']}")
    if manifest["observed_exit_code"] is not None:
        add(f"  observed exit   {manifest['observed_exit_code']}")
    add("")

    add("## What failed")
    failure = manifest["failure"]
    if failure:
        lines.extend(_wrap(f"{failure['exception_type']}: "
                           f"{failure['message']}"))
        add(f"  from the {failure['found_in']} in {failure['source']}")
        if failure["found_in"] == "last product message":
            add("  (this product prints most refusals as one line and exits "
                "2 with")
            add("  no traceback, so this may BE the failure; check the exit "
                "code)")
        elif not failure.get("traceback"):
            add("  (no traceback was recorded with it)")
    else:
        add("  Nothing said it failed: no receipt carried an error, no log")
        add("  carried a traceback, and no line was printed in the product's")
        add("  own voice.  If a command did fail, its output went somewhere")
        add("  this directory does not hold -- pass it with --log.")
    if manifest["warnings"]:
        add("")
        add("  Warnings this run printed and continued past:")
        for warning in manifest["warnings"]:
            wrapped = _wrap(warning, "      ")
            add("    - " + wrapped[0].lstrip())
            lines.extend(wrapped[1:])
        if manifest["warnings_withheld"]:
            add(f"    ... and {manifest['warnings_withheld']} more distinct "
                "warning(s) not listed here.")
    for name in manifest["erased_producer_message_in"]:
        add("")
        add(f"  NOTE: {name} carries an EMPTY 'pipeline producer exited'")
        add("  message.  In this build that is what an out-of-space failure")
        add("  looks like AFTER it has erased its own diagnostics: the")
        add("  decoder's error writes go to the same filesystem that just")
        add("  filled, and the staging directory holding the gate note is")
        add("  removed on unwind.  The empty message is not additional")
        add("  information -- check the Disk section below first.")
    add("")

    add("## Included")
    if manifest["included"]:
        for entry in manifest["included"]:
            add(f"  {entry['path']:<48} {entry['stored_bytes']:>8} B  "
                f"[{entry['category']}] {entry['note']}")
    else:
        add("  (nothing readable: no text artifact was found)")
    add(f"  ... plus a name-and-size inventory of "
        f"{len(manifest['inventory'])} file(s)"
        + (" (TRUNCATED)" if manifest["inventory_truncated"] else "")
        + ", this manifest, and manifest.json")
    add("")
    add("  Content is read only from files this product writes.  Anything")
    add("  else in the run directory is inventoried by name and size and "
        "never")
    add("  opened.")
    refused = manifest["refused_before_reading"]
    if refused:
        add("")
        add(f"  {sum(refused.values())} path(s) were refused before being "
            "opened,")
        add("  read or listed -- not scrubbed afterwards, never looked at:")
        for reason, count in sorted(refused.items()):
            add(f"    {count:>5}  {reason}")
        add("  Their names are not printed here: a file name can itself be")
        add("  the secret.")
    add("")

    add("## Missing")
    missing = manifest["missing_expected_artifacts"]
    route = manifest["route_detected"]
    # Only the artifacts belonging to the route that actually ran are a
    # gap.  Listing a domain-tree receipt as "missing" after a
    # single-domain run is noise, and noise is how a reader learns to
    # skip a section that sometimes matters.  manifest.json keeps the
    # exhaustive list either way.
    relevant = [item for item in missing
                if route is None or item["route"] in (route, _EVERY_ROUTE)]
    other = len(missing) - len(relevant)
    if relevant:
        add("  Each line names an artifact that was NOT found and what "
            "would")
        add("  have been inside it:")
        for item in relevant:
            add(f"  - {item['artifact']}  ({item['route']})")
            for line in _wrap(item["would_have_shown"], "      "):
                add(line)
    elif missing:
        add("  Nothing the detected route writes was absent.")
    else:
        add("  Nothing expected was absent.")
    if other:
        add(f"  ({other} artifact(s) belonging to other routes are also "
            "absent,")
        add("  which is what a route that was not taken looks like; "
            "manifest.json")
        add("  lists them.)")
    if not manifest["route_detected"]:
        add("")
        add("  NO receipt of any route was found here.  Either this is not a")
        add("  run directory, or the run failed before it wrote its first")
        add("  receipt.  This bundle is thin, and that is why.")
    for note in manifest["notes"]:
        add(f"  - {note}")
    add("")

    add("## Disk")
    for label, block in sorted(manifest["volumes"].items()):
        add(f"  {label:<20} {block.get('path')}")
        add(f"  {'':<20} free {_bytes_human(block.get('free_bytes'))} "
            f"of {_bytes_human(block.get('total_bytes'))}")
    if manifest["tight_volumes"]:
        add("  WARNING: " + ", ".join(manifest["tight_volumes"])
            + " is nearly full.  Out of space is")
        add("  the first suspect here, and in this build a full volume "
            "erases")
        add("  its own diagnostics -- so an empty error message is "
            "consistent")
        add("  with it rather than evidence against it.")
    add("")

    add("## Machine")
    machine = manifest["machine"]
    add(f"  os              {machine['os_family']} {machine['os_release']} "
        f"({machine['machine']})")
    add(f"  cpus            {machine['processor_count']}")
    add(f"  python          {machine['python']['python_version']} "
        f"({machine['python']['python_implementation']})")
    for name, version in sorted(machine["python"]["packages"].items()):
        if version:
            add(f"    {name:<14}{version}")
    gpu = machine["gpu"]
    add(f"  gpu probe       {gpu.get('probe')}")
    for device in gpu.get("devices", []):
        memory = device.get("total_memory_bytes")
        add(f"    [{device.get('index')}] {device.get('name')}  "
            + (_bytes_human(memory) if memory
               else str(device.get("total_memory", ""))))
    if gpu.get("nvidia_driver_version"):
        add(f"  nvidia driver   {gpu['nvidia_driver_version']}")
    if gpu.get("driver_version"):
        add(f"  cuda driver api {gpu['driver_version']}")
    if gpu.get("runtime_version"):
        add(f"  cuda runtime    {gpu['runtime_version']}")
    if gpu.get("note"):
        add(f"  note            {gpu['note']}")
    add("")

    add("## What was removed, so this is anonymous")
    add("  Every path, log line, receipt value and file name in this bundle")
    add("  was rewritten to take these out:")
    for name, count in sorted(manifest["redaction"]["classes"].items()):
        add(f"    {name:<24}{count:>6}  "
            f"{manifest['redaction']['descriptions'][name]}")
    add("")
    add(f"  {manifest['environment']['omitted_count']} environment "
        "variable(s) were dropped.  Only these were kept:")
    for name in sorted(manifest["environment"]["collected"]):
        add(f"    {name}={manifest['environment']['collected'][name]}")
    add("")
    add("## What was KEPT on purpose")
    for sentence in manifest["redaction"]["kept_on_purpose"].split("  "):
        if sentence.strip():
            lines.extend(_wrap(sentence))
    add("")

    add("## What this bundle cannot explain")
    for limit in manifest["limits"]:
        wrapped = _wrap(limit, "    ")
        add("  - " + wrapped[0].lstrip())
        lines.extend(wrapped[1:])
    add("")
    add("  Open it before you send it: it is a plain zip and every member "
        "is")
    add("  UTF-8 text you can read.")
    return "\n".join(lines) + "\n"


def _archive_bytes(stamp: str, manifest_text: str, manifest: dict,
                   payloads: list[tuple[str, str]]) -> bytes:
    """The whole archive, in memory.

    In memory because the volume this bundle is ABOUT may be the one that
    is full: a staging tree would fail exactly when the bundle matters
    most.
    """

    root = f"gpuwm-report-{stamp}"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/MANIFEST.txt", manifest_text)
        archive.writestr(f"{root}/manifest.json",
                         json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name, text in payloads:
            archive.writestr(f"{root}/{name}", text)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Writing, with somewhere else to go
# ---------------------------------------------------------------------------

class ReportWriteError(RuntimeError):
    """Every candidate location refused the bundle."""


def _candidate_paths(output, stamp: str) -> list[Path]:
    name = f"gpuwm-report-{stamp}.zip"
    candidates: list[Path] = []
    if output is not None:
        output = Path(output)
        candidates.append(output / name if output.is_dir() else output)
    for directory in (_cwd_or_none(), Path(tempfile.gettempdir()),
                      _home_or_none()):
        if directory is not None:
            candidates.append(directory / name)
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _os_reason(error: OSError) -> str:
    if error.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
        return "no space left on that volume"
    return f"{type(error).__name__}: {error.strerror or error}"


def write_report(report: Report, output=None) -> Report:
    """Write the archive to the first location that accepts it.

    A named ``--output`` is tried first and, when it fails, the failure
    is said out loud and the next candidate is tried: a reporter whose
    disk just filled needs the file more than they need it in the
    directory they picked.  When nothing accepts it, one sentence says so
    and names everything that was tried.
    """

    refusals: list[str] = []
    for candidate in _candidate_paths(output, report.manifest["created_utc"]):
        # Written beside the target and moved into place.  Writing the
        # target directly and unlinking it on failure DESTROYS whatever
        # was already there -- and the first candidate is a path the
        # reporter named, on the volume they said had room, which is
        # exactly where an existing file is most likely to matter.
        partial = candidate.with_name(candidate.name + ".part")
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with open(partial, "wb") as handle:
                handle.write(report.archive)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, candidate)
        except OSError as error:
            refusals.append(f"{candidate}: {_os_reason(error)}")
            try:
                partial.unlink()
            except OSError:
                pass
            continue
        report.written_path = candidate
        report.fallback_reason = refusals[0] if refusals else None
        return report
    raise ReportWriteError(layered(
        "woof report could not write the bundle anywhere.\n"
        + "\n".join(f"  tried {item}" for item in refusals)
        + "\n  remedy: name a volume that has room --\n"
          "    woof report --output /path/on/another/volume",
        "The bundle is assembled in memory precisely so that a full "
        "volume cannot stop it being written somewhere else.  When the "
        "directory you named, the working directory, the system "
        "temporary directory and the home directory ALL refuse the "
        "write, there is no location left to try without being told "
        "one."))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def report_main(args) -> int:
    report = build_report(args.run_dir, exit_code=args.exit_code,
                          extra_logs=tuple(args.extra_logs or ()))
    print(report.text)
    if args.dry_run:
        print("dry run: nothing was written.  Drop --dry-run to write the "
              "bundle.")
        return 0
    write_report(report, args.output)
    if report.fallback_reason:
        warn(f"the first location refused the bundle "
             f"({report.fallback_reason}); it was written to "
             f"{report.written_path} instead",
             why="The archive is built in memory and written once, so a "
                 "volume that is full or read-only costs a relocation "
                 "rather than the whole bundle.  The candidates, in "
                 "order, are: --output, the working directory, the "
                 "system temporary directory, your home directory.")
    print(f"wrote {report.written_path} "
          f"({_bytes_human(len(report.archive))})")
    print("Read it before you send it -- it is a plain zip of UTF-8 text.")
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "report",
        help="collect one anonymous diagnostic bundle for a run -- its "
             "receipts, the failure, the stage logs, this install's "
             "provenance, the card and the free space -- into a readable "
             "zip you can send (--dry-run prints the manifest and writes "
             "nothing)")
    parser.add_argument(
        "run_dir", nargs="?", type=Path, default=Path("."), metavar="RUNDIR",
        help="the run directory to collect from (default: the current "
             "directory, so `woof report` with no arguments inside a run "
             "works; when the current directory holds no receipt but "
             "out/run below it does, that one is read and the manifest "
             "says so)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, metavar="PATH",
        help="where to write the zip: a file path, or a directory to name "
             "it in (default: the current directory, falling back to the "
             "system temporary directory and then your home directory if "
             "a write is refused)")
    parser.add_argument(
        "--dry-run", "--list", dest="dry_run", action="store_true",
        help="print the manifest -- everything that would be included, "
             "redacted and reported missing -- and write nothing")
    parser.add_argument(
        "--exit-code", dest="exit_code", type=int, default=None, metavar="N",
        help="the exit status the failing command returned, recorded in "
             "the manifest (nothing on disk records it)")
    parser.add_argument(
        "--log", dest="extra_logs", action="append", type=Path, default=None,
        metavar="FILE",
        help="an additional log file to include, for output that was "
             "redirected outside the run directory (repeatable)")
    parser.set_defaults(func=report_main)
    return parser


__all__ = [
    "BUNDLE_SCHEMA", "CLASS_DESCRIPTIONS", "Entry", "LIMITS", "PLACEHOLDERS",
    "Redactor", "Report", "ReportWriteError", "build_report",
    "register_cli", "report_main", "write_report",
]
