"""A refused include tree stays a compiler fallback without repeated scans."""
from pathlib import Path
import pytest
from woof.kernel_dependencies import include_tree_dependencies


def test_refused_tree_is_scanned_once_and_stays_refused(tmp_path, monkeypatch):
    root = tmp_path / 'include'; root.mkdir()
    (root / 'escape.h').write_text('#include "../outside.h"\n')
    (tmp_path / 'outside.h').write_text('int outside;\n')
    scans = []
    original = Path.rglob
    def counted(path, pattern):
        scans.append(path)
        return original(path, pattern)
    monkeypatch.setattr(Path, 'rglob', counted)
    snapshots = {}
    for _ in range(3):
        with pytest.raises(ValueError, match='escape include roots'):
            include_tree_dependencies(['-I' + str(root)], snapshots)
    assert scans == [root]
    # A different allowed-root set must be evaluated independently.
    assert len(include_tree_dependencies(['-I' + str(root), '-I' + str(tmp_path)], snapshots)) == 2
    # A fresh compiler session sees repaired contents, not a durable refusal.
    (root / 'escape.h').write_text('int local;\n')
    assert include_tree_dependencies(['-I' + str(root)], {})
