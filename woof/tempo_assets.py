"""Acquire TEMPO's lookup tables, pinned by size and SHA-256.

TEMPO (NCAR/TEMPO, Apache-2.0) reads four lookup tables at init.  The
port's kernels pin TEMPO 9adf9ef (tempo_v3.0.3, the RRFS v2 pin); the
tables are TEMPO's published tempo_v3.1.0 release assets, whose table
builder is unchanged in substance from 9adf9ef: tables built from the pinned
source agree with them to 2.2e-12 relative (compiler rounding), see
docs/TEMPO-PORT.md.  A root holding other bytes is refused without
overwrite, as for the Thompson tables.

Root: ``WOOF_TEMPO_TABLE_ROOT`` or ``~/.woof/tables/tempo-v3``.  Source:
``WOOF_TEMPO_TABLE_SOURCE_ROOT`` (a directory holding the set), else the
release URL base (``WOOF_TEMPO_TABLE_ASSET_URL_BASE`` overrides it).
"""
from __future__ import annotations

import os
from pathlib import Path

from woof import fetch_guard
from woof.core.thompson_contract import TableAsset, validate_table_assets
from woof.table_assets import (
    TableAssetError, classify_assets, fetch_asset_from_dir, fetch_asset_from_url,
)
from woof.tables import table_errors
import sys

TEMPO_TABLE_ROOT_ENV = "WOOF_TEMPO_TABLE_ROOT"
TEMPO_TABLE_SOURCE_ROOT_ENV = "WOOF_TEMPO_TABLE_SOURCE_ROOT"
TEMPO_TABLE_ASSET_URL_BASE_ENV = "WOOF_TEMPO_TABLE_ASSET_URL_BASE"
TEMPO_RELEASE_URL_BASE = "https://github.com/NCAR/TEMPO/releases/download/tempo_v3.1.0"

TEMPO_TABLE_ASSETS = (
    TableAsset("ccn_activate.bin", 35_288,
               "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd"),
    TableAsset("freeze_water_data_tempo_v3", 254_944_800,
               "4624120418c25b9c84865c7ab3e15fc4185437e947f23e308fd5f8f51d19aa54"),
    TableAsset("qr_acr_qg_data_tempo_v3", 674_697_960,
               "537d784eb479ca98c82b0c7f33530b3fff984a8c1181c7b08c5e1f1d5d6ee926"),
    TableAsset("qr_acr_qs_data_tempo_v3", 43_764_192,
               "09e1037a6dd79bb848b344332d9b33e53df15d356c222a3080eed6f5f5882ceb"),
)


def tempo_table_root() -> Path:
    override = os.environ.get(TEMPO_TABLE_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".woof" / "tables" / "tempo-v3"


@table_errors("tempo")
def ensure_tempo_tables(root=None, *, source_dir=None) -> Path:
    """Verify or acquire the complete TEMPO table set under one root lock."""
    root = Path(root) if root is not None else tempo_table_root()
    with fetch_guard.hold("fetch-tables", root):
        _valid, invalid, absent = classify_assets(root, TEMPO_TABLE_ASSETS)
        if invalid:
            raise TableAssetError(
                "TEMPO table root holds different bytes; refusing without "
                "overwrite: " + "; ".join(invalid))
        if not absent:
            return root
        source = source_dir or os.environ.get(TEMPO_TABLE_SOURCE_ROOT_ENV, "").strip() or None
        base = (os.environ.get(TEMPO_TABLE_ASSET_URL_BASE_ENV, "").strip().rstrip("/")
                or TEMPO_RELEASE_URL_BASE)
        root.mkdir(parents=True, exist_ok=True)
        for asset in absent:
            print(f"woof: TEMPO first use: staging {asset.filename} into {root} "
                  "(size + SHA-256 verified before install)", file=sys.stderr, flush=True)
            if source is not None:
                fetch_asset_from_dir(root, asset, Path(source))
            else:
                fetch_asset_from_url(root, asset, base + "/" + asset.filename)
        validate_table_assets(root, TEMPO_TABLE_ASSETS)
        print(f"woof: TEMPO tables at {root} verified (size + SHA-256)", file=sys.stderr)
        return root
