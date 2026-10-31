"""Inspect and stage the pinned tables needed by selected physics.

``status(config)`` is read-only and needs neither CUDA nor a network.
``ensure(config, fetch=False)`` raises a named error for incomplete tables;
the default ``fetch=True`` also uses the existing verified staging routes.
Both return :class:`TableStatus`. Existing scheme-specific APIs remain valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from collections.abc import Mapping

from woof.table_assets import (
    MissingTableAssets, TableAssetError, fetch_asset_from_dir,
)
from woof.core.thompson_contract import TableAsset


class CorruptTableAssets(TableAssetError, ValueError):
    """Pinned coefficients differ; using them would change the selected science."""


@dataclass(frozen=True)
class TableFile:
    """One required file, its scheme family, and its expected identity."""
    scheme: str
    path: Path
    bytes: int
    sha256: str
    relative: str = ""
    canonical_lf: bool = False


@dataclass(frozen=True)
class TableStatus:
    """Partition of required files; ``present`` means size AND SHA-256 match."""
    needed: tuple[TableFile, ...]
    present: tuple[TableFile, ...]
    missing: tuple[TableFile, ...]
    corrupt: tuple[TableFile, ...]

    @property
    def ready(self) -> bool:
        return not (self.missing or self.corrupt)


@lru_cache(maxsize=1)
def _manifest():
    path = Path(__file__).parent / "data/physics_tables.v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema"] != "gpuwm-physics-tables-v1":
        raise TableAssetError("unknown physics table manifest; coefficient identities cannot be checked")
    return tuple(data["files"])


def _get(cfg, name, default=0):
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _families(cfg):
    mp = _get(cfg, "mp_physics")
    mp = 900 if mp == "tempo" else int(mp)
    selected = []
    if mp in (8, 28):
        selected.append("thompson44" if _get(cfg, "thompson_version", "wrf_461") == "wrf_44"
                        else "thompson-fork" if _get(cfg, "thompson_version", "wrf_461") == "wrf_39_noaa"
                        else "thompson28" if mp == 28 else "thompson")
    selected += {5: ["ferrier"], 30: ["fast-sbm"], 50: ["p3"], 900: ["tempo"]}.get(mp, [])
    surface = int(_get(cfg, "sf_surface_physics"))
    selected += {2: ["noah"], 3: ["ruc"], 4: ["noahmp"]}.get(surface, [])
    legacy = _get(cfg, "ra_physics")
    lw = _get(cfg, "ra_lw_physics", -1)
    sw = _get(cfg, "ra_sw_physics", -1)
    lw = legacy if lw < 0 else lw
    sw = legacy if sw < 0 else sw
    variant = _get(cfg, "ra_rrtmg_variant", "rte-rrtmgp")
    if lw == 1:
        selected.append("rrtm")
    for stream, scheme in (("lw", lw), ("sw", sw)):
        if scheme == 4:
            selected.append(("rrtmg-" if variant == "rrtmg_legacy" else "rrtmgp-") + stream)
    if variant == "rrtmg_legacy" and (lw == 4 or sw == 4):
        selected.append("ozone")
    if lw == 5 or sw == 5:
        selected.append("goddard")
    if _get(cfg, "aer_opt") == 1 and sw == 4:
        selected.append("rrtmg-aer1")
    if _get(cfg, "aer_opt") == 2 and sw in (4, 5):
        selected.append("aer-opt2")
    if (_get(cfg, "chem_sets", "") and _get(cfg, "aer_ra_feedback")
            and variant == "rte-rrtmgp" and (lw == 4 or sw == 4)):
        selected.append("rrtmgp-aerosol")
    if _get(cfg, "cu_physics") == 1:
        selected.append("kf")
    from woof.config import CCPP_TKE_EDMF_PBL_SCHEME
    if _get(cfg, "bl_pbl_physics") == CCPP_TKE_EDMF_PBL_SCHEME:
        selected.append("satmedmf")
    if _get(cfg, "sf_urban_physics"):
        urban = "urban-lcz" if _get(cfg, "use_wudapt_lcz") else "urban"
        if _get(cfg, "urban_ucm_generation", "wrf_471") == "wrf_480":
            urban += "-480"
        selected.append(urban)
    if _get(cfg, "ghg_input") and (lw in (1, 4, 5) or sw in (4, 5)):
        selected.append("camtr")
    return selected


def _requests(config_or_schemes):
    if hasattr(config_or_schemes, "domains"):
        requests = [(family, domain.run) for domain in config_or_schemes.domains
                    for family in _families(domain.run)]
        # Real-data landuse_init reads these even for an mp-only forecast.
        # A hand-built RunConfig/DomainState has no such preparation seam.
        if config_or_schemes.domains:
            requests += [("landuse", {}), ("noah", {})]
        return requests
    if isinstance(config_or_schemes, str):
        config_or_schemes = [config_or_schemes]
    if isinstance(config_or_schemes, (list, tuple, set, frozenset)):
        names = []
        for family in config_or_schemes:
            names.extend({"rrtmg": ["rrtmg-lw", "rrtmg-sw"],
                          "rrtmgp": ["rrtmgp-lw", "rrtmgp-sw"],
                          "nssl": [], "morrison": []}.get(family, [family]))
        return [(family, {}) for family in names]
    return [(family, config_or_schemes) for family in _families(config_or_schemes)]


def _packaged_path(relative):
    from woof import data_assets
    return Path(data_assets.data_path(relative))


def _user_path(relative):
    return Path.home() / ".woof/tables/packaged" / relative


def _resolved_path(relative, original=None):
    override = os.environ.get("WOOF_TABLE_ROOT", "").strip()
    if override:
        return Path(override) / relative
    packaged = _packaged_path(relative)
    if original is not None and Path(original) != packaged:
        return Path(original)
    # Keep existing package paths. Only a missing packaged file uses a staged copy.
    if packaged.exists() or not _user_path(relative).exists():
        return packaged
    return _user_path(relative)


def _files(family, cfg):
    if family == "thompson44":
        from woof.core.thompson_column import GENERATION_ASSETS, AUXILIARY_ASSET, auxiliary_path, table_root
        root = table_root(cfg)
        return ([TableFile(family, root / name, size, digest)
                 for name, size, digest in GENERATION_ASSETS["wrf_44"]]
                + [TableFile(family, auxiliary_path(), AUXILIARY_ASSET[1], AUXILIARY_ASSET[2])])
    if family in ("thompson", "thompson28", "thompson-fork", "tempo", "p3"):
        if family.startswith("thompson"):
            from woof.physics_compat import thompson_table_root
            from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS, FORK_TABLE_ASSETS
            from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS, AEROSOL_TABLE_PATH_ENV
            root = Path(thompson_table_root())
            assets = CLASSIC_TABLE_ASSETS
            if family == "thompson-fork":
                from woof.physics_compat import thompson_fork_table_root
                root, assets = Path(thompson_fork_table_root()), FORK_TABLE_ASSETS
            result = [TableFile(family, root / a.filename, a.bytes, a.sha256) for a in assets]
            if family == "thompson28":
                result += [TableFile(family, Path(os.environ.get(AEROSOL_TABLE_PATH_ENV) or root / a.filename), a.bytes, a.sha256)
                           for a in AEROSOL_TABLE_ASSETS]
            return result
        if family == "tempo":
            from woof.tempo_assets import TEMPO_TABLE_ASSETS, tempo_table_root
            return [TableFile(family, tempo_table_root() / a.filename, a.bytes, a.sha256)
                    for a in TEMPO_TABLE_ASSETS]
        from woof.core.p3_tables import TABLE_1_2MOM_ASSET, p3_table_root
        a = TABLE_1_2MOM_ASSET
        return [TableFile(family, Path(p3_table_root()) / a.filename, a.size, a.sha256,
                          "p3/tables/" + a.filename)]
    rows = [row for row in _manifest() if row["family"] == family]
    if not rows:
        raise ValueError(f"unknown table family {family!r}")
    files = []
    for row in rows:
        relative = row["relative"]
        original = None
        if family.startswith("urban") and _get(cfg, "urban_table_dir", "") and relative.endswith(".TBL"):
            original = Path(_get(cfg, "urban_table_dir")) / Path(relative).name
        if row.get("generated"):
            from woof.table_assets import generated_table_root
            path = generated_table_root(family) / Path(relative).name
        else:
            path = _resolved_path(relative, original)
        files.append(TableFile(family, path, row["bytes"],
                               row["sha256"], relative, row["canonical_lf"]))
    return files


def _valid(item):
    # Text tables already accepted by the RUC loader retain CRLF/LF equivalence.
    if item.canonical_lf:
        raw = item.path.read_bytes().replace(b"\r\n", b"\n")
        return len(raw) == item.bytes and hashlib.sha256(raw).hexdigest() == item.sha256
    if item.path.stat().st_size != item.bytes:
        return False
    # Inspection must see current bytes, including a same-size copy that
    # preserves mtime. Windows stat change time can be the creation time,
    # so a metadata-only digest cache is insufficient for this public check.
    digest = hashlib.sha256()
    with item.path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == item.sha256


_LOAD_LOCK = threading.RLock()
_LOAD_VERIFIED = set()


def _file_state(path):
    """Replacement-sensitive identity for the process-local load cache."""
    resolved = Path(path).resolve()
    st = resolved.stat()
    return (str(resolved), st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev,
            st.st_ctime_ns)


def _valid_for_load(item):
    # Repeated admission/identity checks otherwise hash large lookup tables on
    # the stepping thread, including every generated-Thompson step. Inspection
    # deliberately bypasses this cache; unchanged loader bytes are verified once.
    with _LOAD_LOCK:
        before = _file_state(item.path)
        key = (before, item.bytes, item.sha256, item.canonical_lf)
        if key in _LOAD_VERIFIED:
            return True
        if item.canonical_lf:
            valid = _valid(item)
        else:
            from woof.file_digest import sha256_file
            valid = (before[1] == item.bytes
                     and sha256_file(before[0]) == item.sha256)
        if valid and _file_state(item.path) == before:
            _LOAD_VERIFIED.add(key)
            return True
        return False


def _status(config_or_schemes, *, for_load=False) -> TableStatus:
    files = {}
    for family, cfg in _requests(config_or_schemes):
        for item in _files(family, cfg):
            files.setdefault((item.path, item.bytes, item.sha256, item.canonical_lf), item)
    needed = tuple(files.values())
    present, missing, corrupt = [], [], []
    validate = _valid_for_load if for_load else _valid
    for item in needed:
        try:
            if not item.path.is_file():
                missing.append(item)
            elif validate(item):
                present.append(item)
            else:
                corrupt.append(item)
        except OSError:
            missing.append(item)
    return TableStatus(needed, tuple(present), tuple(missing), tuple(corrupt))


def status(config_or_schemes) -> TableStatus:
    """Inspect a RunConfig, ExperimentConfig, physics mapping, or family names.

    For example ``status({"mp_physics": 8})`` or ``status(["p3", "noah"])``.
    No fetch, mutation, scheme substitution, or CUDA initialization occurs.
    ``missing`` includes unreadable files; ``corrupt`` includes wrong pins.
    """
    return _status(config_or_schemes)


def _remedy(item):
    return f"woof fetch-tables --schemes {item.scheme} --from DIR --replace-damaged"


def table_errors(family):
    """Keep a loader's established exception type and add its staging command."""
    from functools import wraps
    def decorate(load):
        @wraps(load)
        def checked(*args, **kwargs):
            try:
                return load(*args, **kwargs)
            except (OSError, ValueError, RuntimeError) as error:
                command = f"woof fetch-tables --schemes {family} --from DIR --replace-damaged"
                if command in str(error):
                    raise
                detail = str(error).replace(
                    "woof fetch-tables --schemes thompson --from DIR --replace-damaged", command)
                detail = detail.replace("woof fetch-tables --from DIR", command)
                message = f"{family}: {detail}"
                if command not in detail:
                    message += f"; run `{command}` with the pinned files"
                raise type(error)(message) from error
        return checked
    return decorate


def _refuse(report):
    if report.corrupt:
        item = report.corrupt[0]
        identity = "canonical identity mismatch" if item.canonical_lf else "identity mismatch"
        if item.scheme == "rrtm":
            identity += " (coefficient or record marker mismatch)"
        raise CorruptTableAssets(
            f"{item.scheme} needs {item.path}: corrupt physics table, {identity} "
            f"(expected {item.bytes} bytes, SHA-256 {item.sha256}); "
            f"refusing altered lookup coefficients. Run `{_remedy(item)}` "
            "with a directory holding the pinned files.")
    if report.missing:
        item = report.missing[0]
        companion = ""
        if item.relative:
            from woof import data_assets
            if any(item.relative == tree or item.relative.startswith(tree + "/")
                   for tree in data_assets.COMPANION_TREES):
                companion = (f" Packaged copies come from {data_assets.COMPANION_DISTRIBUTION}; "
                             "repair an incomplete installation with `"
                             + data_assets.companion_reinstall_command() + "`.")
        raise MissingTableAssets(
            f"{item.scheme} needs missing or unreadable physics table {item.path}; "
            f"refusing to advance the selected scheme without coefficients. Run `{_remedy(item)}` "
            "with a directory holding the pinned files." + companion)


def ensure(config_or_schemes, fetch: bool = True) -> TableStatus:
    """Return verified requirements, or a named missing/corrupt/fetch error.

    ``fetch=False`` only checks. With ``fetch=True`` Thompson and TEMPO keep
    their existing first-use staging; other packaged tables can be copied from
    verified installed data into an explicitly selected empty table root.
    Corrupt data is always refused, never interpreted or replaced implicitly.
    """
    return _ensure(config_or_schemes, fetch=fetch)


def _ensure_for_load(config_or_schemes, fetch=True):
    """Internal admission: reuse pins only while the resolved file is unchanged."""
    return _ensure(config_or_schemes, fetch=fetch, for_load=True)


def _ensure(config_or_schemes, *, fetch=True, for_load=False):
    report = _status(config_or_schemes, for_load=for_load)
    deferred = set()
    if fetch and report.missing and not report.corrupt:
        families = list(dict.fromkeys(item.scheme for item in report.missing))
        for family in families:
            if family in ("thompson", "thompson28"):
                from woof.table_assets import require_thompson_tables
                from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS
                from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS
                assets = CLASSIC_TABLE_ASSETS + (AEROSOL_TABLE_ASSETS if family == "thompson28" else ())
                require_thompson_tables(assets=assets)
            elif family == "tempo":
                from woof.tempo_assets import ensure_tempo_tables
                ensure_tempo_tables()
            elif family == "thompson-fork":
                from woof.thompson_fork_assets import stage_thompson_fork_tables
                stage_thompson_fork_tables(raise_errors=True)
            elif family == "satmedmf" and all(
                    item.path.name == "fpvs-f64.bin" for item in report.missing if item.scheme == family):
                # Preserve its existing GPU generator, which verifies the exact
                # output pin before any saturation pressures reach physics.
                from woof.core.satmedmf import _table
                import cupy as cp
                # A host stand-in for CuPy (the CPU field census, host parity
                # backends) has no device to generate on; the scheme builds
                # the table at its first device use, as before this check.
                if hasattr(cp.cuda, "Device"):
                    _table()
                else:
                    deferred.add(family)
            else:
                for item in report.missing:
                    if item.scheme == family and item.relative:
                        mirror = os.environ.get("WOOF_TABLE_ASSET_URL_BASE", "").strip()
                        if mirror:
                            from woof.table_assets import _fetch_asset_from_url
                            _fetch_asset_from_url(item.path.parent,
                                                 TableAsset(item.path.name, item.bytes, item.sha256),
                                                 mirror.rstrip("/") + "/" + item.relative,
                                                 offline_command=_remedy(item))
                            continue
                        source = _packaged_path(item.relative)
                        if source != item.path and source.is_file():
                            _copy(item, source, False)
        report = _status(config_or_schemes, for_load=for_load)
        if deferred:
            from dataclasses import replace as _replace
            report = _replace(report, missing=tuple(
                item for item in report.missing if item.scheme not in deferred))
    _refuse(report)
    return report


def _copy(item, source, replace):
    # Verify exactly the source bytes that will be installed, including text
    # normalization permitted by the existing table contract.
    item.path.parent.mkdir(parents=True, exist_ok=True)
    if item.canonical_lf:
        from woof import fetch_guard
        from woof.table_assets import _stage, _staging_path
        with fetch_guard.hold("fetch-tables", item.path.parent):
            if item.path.exists() and not replace:
                _refuse(TableStatus((item,), (), (), (item,)))
            raw = source.read_bytes().replace(b"\r\n", b"\n")
            item.path.parent.mkdir(parents=True, exist_ok=True)
            asset = TableAsset(item.path.name, item.bytes, item.sha256)
            temp = _staging_path(item.path.parent, asset)
            try:
                temp.write_bytes(raw)
                _stage(temp, item.path, asset)
            finally:
                temp.unlink(missing_ok=True)
    else:
        fetch_asset_from_dir(item.path.parent,
                             TableAsset(item.path.name, item.bytes, item.sha256),
                             source.parent, replace_damaged=replace)
    print(f"woof fetch-tables: {item.scheme}: verified and installed {item.path}", file=sys.stderr)


def stage_schemes(schemes, *, source_dir=None, replace_damaged=False):
    """CLI staging leg, using the same pins and atomic installer as Thompson."""
    if source_dir is None:
        return ensure(schemes)
    report = status(schemes)
    if report.corrupt and not replace_damaged:
        _refuse(report)
    for item in (*report.missing, *report.corrupt):
        source_root = Path(source_dir)
        candidates = [source_root / item.relative] if item.relative else []
        candidates += [source_root / item.scheme / item.path.name, source_root / item.path.name]
        source = next((p for p in candidates if p.is_file()), candidates[-1])
        if not source.is_file():
            raise MissingTableAssets(f"{item.scheme} needs {item.path.name}: absent from --from directory {source_root}")
        _copy(item, source, replace_damaged)
    report = status(schemes)
    _refuse(report)
    return report


def table_path(family, path):
    """Resolve and pin-check a packaged table before its loader interprets it."""
    path = Path(path)
    family_rows = [r for r in _manifest() if r["family"] == family]
    rows = [r for r in family_rows if Path(r["relative"]).name == path.name]
    if not rows and len(family_rows) == 1:
        rows = family_rows  # a loader may accept an explicitly named mirror file
    if len(rows) != 1:
        raise TableAssetError(f"{family}: {path.name} has no unique physics table pin")
    row = rows[0]
    resolved = _resolved_path(row["relative"], path)
    item = TableFile(family, resolved, row["bytes"], row["sha256"], row["relative"], row["canonical_lf"])
    try:
        if not resolved.is_file():
            _refuse(TableStatus((item,), (), (item,), ()))
        if not _valid_for_load(item):
            _refuse(TableStatus((item,), (), (), (item,)))
    except OSError as exc:
        raise MissingTableAssets(f"{family} cannot read {resolved}: {exc}; run `{_remedy(item)}`") from exc
    return resolved


__all__ = ["TableFile", "TableStatus", "MissingTableAssets", "TableAssetError",
           "CorruptTableAssets", "status", "ensure"]
