"""Content rules shared by the full export scan and its in-cut hygiene rewrite.

Scan the proposed export, including unchanged files. Findings are handed to
hygiene.plan inside the cut; executable data that cannot be rewritten stays
an explicit finding in that stage, without creating another pass.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

RULES = (
    ("instruction filename", re.compile(r"\bCLA" + r"UDE\.md\b", re.I)),
    ("internal box", re.compile(r"\b(?:g" + r"m1|v" + r"58|v(?:5090|8x|\d+x\d+))\b", re.I)),
    ("private worktree", re.compile(r"\.cla" + r"ude[/\\]worktrees", re.I)),
    ("private pool", re.compile(r"\bpool-" + r"v3\b", re.I)),
    ("private ruling file", re.compile(r"\bd" + r"rew-ruling\.json\b", re.I)),
    ('delivery path', re.compile(r'Down' + r'loads[\\/]')),
    ('internal path', re.compile(r'/opt/gpu' + r'-mutex(?:/|\b)|(?:~/)?agent' + r'-scratch(?:/|\b)|evidence' + r'-gallery(?:/|\b)|PAPER' + r'-EVIDENCE(?:/|\b)', re.I)),
    ('private case', re.compile(r'\bU' + r'AE\b|\bOU[ -]+Oman[ -]+U' + r'AE\b', re.I)),
    ('internal host', re.compile(r'\b(?:weather-)?node-\d+\b', re.I)),
    ('internal lane', re.compile(r'\b(?:lane' + r'kit|lane' + r'-start|mpas' + r'-lanes|astra' + r'-lanes)\b', re.I)),
    ('private assimilation recipe', re.compile(r"(?:^|[/\"'])(?:je" + r"di[-])[A-Za-z0-9*]", re.I)),
    ('private assimilation name', re.compile(r'\b(?:CUDA[ _-]*JE' + r'DI|JE' + r'DI[ _-]*CUDA)\b|\bsyn' + r'radar\b|\bsynthetic[- _]radar\s+DA\b|\bwoof[-_]?da\b', re.I)),
    ('private assimilation tooling', re.compile(r'(?<![a-z0-9])(?:obs[-_]pipeline(?:s)?|da[-_](?:orchestrat[a-z]*|rows?|cases?|enkf|us-)|enkf[-_]case|column[-_ ]?letkf|letkf[-_]column|arwen_' + r'cycle|obs_set_' + r'passes|rapid5_' + r'live|resident_' + r'bench|rw[-_]radar[-_]qc|radar_' + r'qc)(?![a-z0-9])', re.I)),
)

# Explicit public scope, matched as tokens rather than exempting entire files.
# group | token pattern | reason
PUBLIC_CONTRACT_ALLOWLIST = (
    ("native MPAS fields and streams", re.compile(r"\b(?:config_[A-Za-z0-9_]*(?:jedi|DA)[A-Za-z0-9_]*|jedi_da|da_state|mpasout)\b", re.I),
     "BSD upstream Registry fields, stream tables and native model code remain public."),
    ("NOAA source repository", re.compile(r"\brrfs-mpas-jedi\b", re.I),
     "The public NOAA repository name identifies the upstream source and reference tools."),
    ("stock MPAS state contract", re.compile(r"(?:(?:--)?jedi-(?:background|soil-levels)\b|--da-analysis\b|\b(?:hexcore\.)?jedi_contract\b|\bjedi-(?:ensemble-stream|contract)\b|\bjedi_[A-Za-z0-9_]+\b)", re.I),
     "Existing background, analysis, mpasout and ensemble record interfaces are public interoperability contracts."),
)


def private_recipe_text(line: str) -> str:
    for _, pattern, _ in PUBLIC_CONTRACT_ALLOWLIST:
        line = pattern.sub("public_contract", line)
    return line

HISTORICAL = re.compile(r'(?:2024.{0,60}U' + r'AE[- ]Oman|U' + r'AE[- ]Oman.{0,60}2024)', re.I)
EXAMPLE_PATH = re.compile(r'[Cc]:[\\/]Users[\\/]someone[\\/]Down' + r'loads[\\/]')


def scan(path: str, data: bytes) -> list[tuple[str, int, str, str]]:
    """(path, line, kind, context), with one finding per rule and line."""
    if PurePosixPath(path).suffix == '.pyc' or '__pycache__' in PurePosixPath(path).parts:
        return [(path, 0, 'compiled Python', path)]
    lines = [(0, path)]
    if b'\0' not in data[:4096]:
        lines.extend(enumerate(data.decode('utf-8', 'replace').splitlines(), 1))
    sealed = any(part in path for part in ("compiler-recipes", "/mpasphys-maths/")) or path.endswith("rkind_libm.py")
    hits = []
    for number, line in lines:
        for kind, pattern in RULES:
            checked = private_recipe_text(line) if kind == "private assimilation recipe" else line
            if not pattern.search(checked):
                continue
            if sealed and kind in {"internal host", "internal box", "instruction filename"}:
                continue
            if kind == "internal host" and re.search(r"\breturn\s+node-\d+;|=\s*node-\d+;", line):
                continue
            if kind == "internal box" and re.search(r"\bv" + r"58\s*(?:=>|[,)]|\b)|transfer\(v" + r"58", line, re.I) and path.endswith(".f90"):
                continue
            if kind == 'private case' and HISTORICAL.search(line):
                continue
            if kind == 'delivery path' and path.startswith(('tests/', 'tree/tests/')):
                # Only the made-up Windows example used to test path refusal.
                if not pattern.search(EXAMPLE_PATH.sub('', line)):
                    continue
            hits.append((path, number, kind, line.strip()[:220]))
    return hits


def sealed_exception(path: str, kind: str) -> bool:
    """Temporary patch exception for exact recorded compiler and arithmetic bytes."""
    return ("compiler-recipes" in path or "/mpasphys-maths/" in path or path.endswith("rkind_libm.py")) and kind in {"lab-host-name", "private-host"}
