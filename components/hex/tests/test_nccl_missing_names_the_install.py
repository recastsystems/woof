"""A host with no NCCL is told the one pip line that gives it one.

THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026): `pip install "recast-woof[gpu]"` installs no NCCL (cupy's
toolkit extra carries none), and `woof hex forecast --hosts ...` needs it on every host. On a fresh install the
refusal said only "NCCL/RCCL is not installed ... WOOF_HEX_NCCL_LIBRARY may name an installed library", which names
no package; the pip wheel nvidia-nccl-cu12 is the one hex already looks inside (nvidia.nccl), and installing it made
the two-host run go.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import nccl_runtime  # noqa: E402


@pytest.fixture
def no_library(monkeypatch, tmp_path):
    monkeypatch.setattr(nccl_runtime, "_LIBRARY", None)
    monkeypatch.setattr(nccl_runtime, "_LIBRARY_PATH", None)
    monkeypatch.setattr(nccl_runtime, "_candidates", lambda: [str(tmp_path / "libnccl.so.2")])


def test_nvidia_refusal_names_the_pip_wheel(no_library, monkeypatch) -> None:
    import woof.gpu_smi as smi

    monkeypatch.setattr(smi, "is_amd", lambda: False)
    with pytest.raises(nccl_runtime.NcclError) as caught:
        nccl_runtime.library()
    text = str(caught.value)
    assert "pip install nvidia-nccl-cu12" in text
    assert "every host" in text
