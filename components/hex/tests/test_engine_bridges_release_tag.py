"""A pinned engine bridge or source prebuilt whose bytes changed after its release was public names its own release.

THE BREAKAGE THIS PREVENTS (hex 0.6.0 candidate, 16 Nov 2026): the public release engine-bridges-v0.4.12 holds the
rw_mpas_convert that hex 0.5.0 pins (sha256 c38b6f95..., 4,345,616 bytes). 0.6.0 pinned the converter rebuilt from woof
f717cd5e2c (092b26d4..., 4,197,416 bytes) under the same tag, so the release build's prebuilt step (publish.yml) would
have fetched the old file and stopped the 0.6.0 release ("is not the pinned sha256") after its tag and release page
existed; replacing the public file would break 0.5.0's own pin. The row now names engine-bridges-v0.6.0, and every
fetch of a bridge or prebuilt (publish.yml, setup.py, tools/stage_release_assets.py) takes a row's release_tag.
"""
import json
import re
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
PINS = json.loads((TREE / "../../woof/hex/data/engine-bridges.json").read_text(encoding="utf-8"))
# the bytes the public releases hold under the shared tag, by name (read 2026-11-16 from the public release)
PUBLIC_AT_SHARED_TAG = {"rw_mpas_convert": "c38b6f950e7fc697"}


def test_a_rebuilt_prebuilt_names_its_own_release():
    rows = dict(PINS["bridges"])
    rows.update(PINS.get("source_prebuilts", {}).get("files", {}))
    for name, public in PUBLIC_AT_SHARED_TAG.items():
        row = rows[name]
        if row["sha256"].startswith(public):
            continue
        tag = row.get("release_tag", PINS["release_assets"]["tag"])
        assert tag != PINS["release_assets"]["tag"], (
            f"{name}: pinned {row['sha256'][:16]} under {tag}, which publicly holds {public}; name its own release")


def test_every_fetch_of_a_bridge_takes_the_rows_release_tag():
    if not (TREE / "setup.py").is_file():
        # (WOOF 1.1.7) hex's setup.py and publish workflow are not folded: WOOF takes the hex helpers from the
        # released gpuwm-hex wheel (stage_hex_helpers.py) and fetches no bridge by a row's tag
        import pytest
        pytest.skip("hex's setup.py is not folded into WOOF: its helpers come from the released gpuwm-hex wheel, "
                    "never from a download a row's release_tag names; hex's own battery checks its fetches")
    setup = (TREE / "setup.py").read_text(encoding="utf-8")
    assert "row.get('release_tag', assets['tag'])" in setup
    workflow = TREE.parent / ".github/workflows/publish.yml"
    if workflow.exists():   # the source distribution carries tree/ only
        text = workflow.read_text(encoding="utf-8")
        urls = re.findall(r"releases/download/\{[^}]*\}", text)
        assert urls and all("release_tag" in u for u in urls), urls
