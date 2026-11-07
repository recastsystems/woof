"""Table-driven Earth2Studio acquisition and its bound preparation handoff.

The runtime stays in WOOF_EARTH2_PYTHON. Models, loader arguments, licences,
profiles and donor policy are data, shared by fetch and the go launch route.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

# netCDF4's HDF5 build is not thread-safe; the ensemble door prepares members on threads of one
# process, and concurrent opens failed with "NetCDF: Can't open HDF5 attribute".
_HDF5_LOCK = threading.Lock()

TABLE = Path(__file__).with_name("authorities") / "earth2-models.v1.json"
PYTHON_ENV = "WOOF_EARTH2_PYTHON"
RECEIPT = "earth2-run.json"
PREP_HANDOFF_SCHEMA = "gpuwm-fetch-prep-arguments-v1"
PUBLISHES_PREP_HANDOFF = True
FETCH_ACCEPTS_AREA = True


def model_table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def resolve_donor_policy(row: dict, cycle: datetime, mode: str = "auto", *, now=None) -> dict:
    """Resolve one same-cycle physical donor and its sealed profile from table data.

    ``forecast`` is explicit even for an archived forecast replay. ``auto``
    uses the table's age boundary; a row may override either mode or name
    one donor for both. Acquisition and model inference remain runner-owned.
    """
    table = model_table()
    policy = table["donor_policy"]
    if mode == "auto":
        current = now or datetime.now(cycle.tzinfo)
        mode = ("hindcast" if cycle < current - timedelta(days=policy["auto_hindcast_age_days"])
                else "forecast")
    if mode not in policy["defaults"]:
        raise ValueError(f"donor mode {mode!r} is not declared in donor_policy.defaults")
    override = row.get(policy["row_override_key"])
    if override is None:
        override = next((rule["donor"] for rule in policy.get("row_defaults", ())
                         if all(row.get(key) == value for key, value in rule["match"].items())), {})
    donor_id = (override if isinstance(override, str) else
                override.get(mode, override.get("source", policy["defaults"][mode])))
    donor = dict(policy["donors"][donor_id])
    profile = row["profile"]
    variants = policy["profile_variants"].get(profile)
    if variants is None:
        variants = next((values for values in policy["profile_variants"].values()
                         if profile in values.values()), None)
    if variants is None or donor_id not in variants:
        raise ValueError(f"profile {profile!r} has no declared {donor_id!r} donor variant")
    return {**donor, "id": donor_id, "mode": mode, "profile": variants[donor_id],
            "cycle": cycle.isoformat(), "hours": 0}


def source_aliases() -> tuple[str, ...]:
    """Unclaimed runnable model shorthands from the table, without I/O."""
    aliases = []
    for key, row in model_table()["models"].items():
        if row.get("kind") == "diagnostic":
            continue
        try:
            resolve_model(key)
        except ValueError:
            continue
        aliases.extend((key, *row.get("aliases", ())))
    return tuple(aliases)


def acquisition_hints(source: str, model: str | None = None) -> dict:
    """Pure planning hook; never imports a model runtime or downloads bytes."""
    named = None if source in ("earth2", "earth2studio") else model_declaration(source)[0]
    if model is not None and named is not None and model_declaration(model)[0] != named:
        raise ValueError("--model conflicts with the model named by --source")
    selected = model or named
    return {"model": model_declaration(selected)[0]} if selected is not None else {}


def model_declaration(name: str | None) -> tuple[str, dict]:
    """Read identity for planning; acquisition separately enforces runtime policy."""
    table = model_table()
    models = table["models"]
    wanted = (name or "").strip().lower().replace("_", "-")
    if wanted in ("", "earth2", "earth2studio"):
        wanted = table["runtime"].get("default_model", "")
    if wanted.startswith("earth2-"):
        wanted = wanted[len("earth2-"):]
    for model_id, row in models.items():
        if wanted == model_id or wanted in (alias.lower() for alias in row.get("aliases", ())):
            return model_id, row
    raise ValueError("choose a model row: " + ", ".join(sorted(models)))


def resolve_model(name: str | None) -> tuple[str, dict]:
    model_id, row = model_declaration(name)
    from woof.earth2_licensing import require_licence
    require_licence(model_id, row)
    if not row.get("profile"):
        raise ValueError(f"{model_id}: {row.get('needs', 'no packaged profile')}")
    refusal = row.get("runtime_refusal") or row.get("refusal") or row.get("licence_refusal")
    if refusal:
        raise ValueError(f"{model_id}: {refusal}")
    weights = row.get("weights") or {}
    if weights.get("account_needed") and weights.get("host") != "huggingface":
        raise ValueError(f"{model_id}: weights need an account at {weights.get('host')}; "
                         "only anonymous public weights are acquired")
    return model_id, row


def acquisition_facts(source: str) -> dict:
    """The inventory and readiness doors describe the same table policy."""
    from woof.source_adapters import get_source_adapter

    adapter = get_source_adapter(source)
    return {"generation": "generated on the card at run time",
            "model": adapter.model_key or model_table()["runtime"]["default_model"],
            "readiness_basis": "the same-cycle donor analysis readiness",
            "donor_policy": model_table()["donor_policy"]}


def readiness(source, cycle, hours, *, now=None, **options):
    """Ask for the donor analysis, without loading models or acquiring bytes."""
    from woof import fetch, source_readiness as readiness_api
    from woof.source_adapters import get_source_adapter

    checked = readiness_api._naive(now) if now else readiness_api._utc_now()
    named_cycle = None
    try:
        adapter = get_source_adapter(source)
        model, row = resolve_model(adapter.model_key)
        named_cycle = (fetch.parse_cycle(cycle, source)
                       if isinstance(cycle, str) and cycle != "latest" else cycle)
        donor = resolve_donor_policy(
            row, checked if named_cycle == "latest" else named_cycle, now=checked)
    except ValueError as error:
        return (readiness_api._refusal_document(
            source, named_cycle if isinstance(named_cycle, datetime) else None,
            "named", hours, options.get("as_posted", True), str(error), checked),
            readiness_api.REFUSED_EXIT)
    # The donor is an analysis at lead zero, not an AI-member forecast
    # window. In particular, the AI member number is not a donor member.
    donor_document, code = readiness_api.readiness(
        donor["source"], named_cycle, donor["hours"], start_hour=0,
        provider=donor["provider"], transport=options.get("transport"),
        no_probe=options.get("no_probe", False), probe=options.get("probe"),
        as_posted=options.get("as_posted", True), now=checked)
    document = {**donor_document, "source": source,
                "member": options.get("member"),
                "window": {"start_lead": options.get("start_hour") or 0,
                           "hours": hours, "cadence": options.get("cadence"),
                           "final_lead": (options.get("start_hour") or 0) + hours},
                "acquisition_owner": __name__,
                "generation": {"description": "generated on the card at run time",
                               "model": model, "donor": donor,
                               "readiness": donor_document}}
    return document, code


def require_inference_vram(model_id, row):
    """Check a row's measured capacity floor only when generating new output."""
    minimum = float(row.get("min_vram_gib", 0))
    if minimum <= 0:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        raise ValueError(f"{model_id}: inference needs {minimum:g} GiB of GPU memory; CUDA devices are hidden")
    selector = visible.split(",")[0] if visible else "0"
    command = [*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=memory.total,memory.free",
               "--format=csv,noheader,nounits", "--id=" + selector]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        lines = result.stdout.strip().splitlines()
        if len(lines) != 1:
            raise ValueError("expected one selected GPU")
        total, free = (float(v.strip()) for v in lines[0].split(","))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError(f"{model_id}: cannot verify the row's {minimum:g} GiB memory requirement: {error}") from error
    measured = dict(minimum_gib=minimum, total_mib=total, free_mib=free, command=command,
                    note=row.get("vram_note"))
    if min(total, free) < minimum * 1024:
        raise ValueError(f"{model_id}: inference needs {minimum:g} GiB of free GPU memory; "
                         f"card has {total/1024:.2f} GiB total, {free/1024:.2f} GiB free. "
                         + row.get("vram_note", "The measured table requirement is not met."))
    return measured


def earth2_python() -> str:
    """Require the table's version, before any model or donor download."""
    # WOOF_EARTH2_PYTHON, then the managed environment `woof setup --with-earth2` builds, then
    # this interpreter (woof.earth2_env). Model extras that need source builds are not built here.
    from woof import earth2_env
    runtime = model_table()["runtime"]
    found, notes = earth2_env.find_python(exact=True)
    if found:
        return found
    raise ValueError(f"no interpreter has earth2studio=={runtime['pinned_version']} ({'; '.join(notes)}). "
                     f"Run `{earth2_env.SETUP_COMMAND}`, or use runtime.install_recipes in {TABLE} and "
                     f"set {PYTHON_ENV} to that venv's python")


def members_from_flag(member) -> list[int]:
    raw = [0] if member is None else (member if isinstance(member, (tuple, list)) else str(member).split(","))
    try:
        values = [int(value) for value in raw]
    except (ValueError, TypeError):
        raise ValueError("--member must be a nonnegative integer or comma-separated member list") from None
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("--member needs distinct nonnegative IDs")
    return values


def _donor(*, table, row, cycle, area, out, force):
    """A donor acquisition owner may supply the policy's bound supplements."""
    from importlib import import_module
    policy = table.get("donor_policy", {})
    owner = policy.get("acquisition_owner")
    if owner:
        return import_module(owner).acquire(row=row, cycle=cycle, area=area, out=out,
                                            force=force, policy=policy)
    selected = resolve_donor_policy(row, cycle)
    workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    os.environ.setdefault("GPUWM_RW_ZARR_WORKERS", str(workers))
    analysis = out / "analysis"
    # The default donor is the whole globe: a full 360-degree band.  Named
    # breakage: the old default east edge 179.75 gave a 359.75-degree span,
    # which fetch.parse_area reads as the antimeridian complement, so the
    # donor came back as a 0.25-degree strip (179.5..180.25) and the storm
    # nest prep refused it as cross-grid against the 0..359.75 FCN3 grid.
    south, west, north, east = area or (-90, -180, 90, 180)
    # The public reader accepts signed longitudes, while E2S crops in 0..360.
    # A full band stays -180..180 (span 360, never swapped); otherwise the
    # west edge wraps into [-180, 180) and the east edge into (-180, 180],
    # so an east edge on the antimeridian stays +180 instead of folding
    # onto the west edge.
    from woof.fetch import _wrap_lon, _wrap_lon_east
    if float(east) - float(west) >= 360.0:
        signed_west, signed_east = -180.0, 180.0
    else:
        signed_west, signed_east = _wrap_lon(float(west)), _wrap_lon_east(float(east))
    signed_area = f"{south},{signed_west},{north},{signed_east}"
    recipe = table["runtime"]["donor_acquisition"][selected["provider"]]
    if recipe["kind"] == "arco":
        from woof.era5_arco import retrieve_era5_arco
        retrieve_era5_arco(cycle=cycle, hours=0,
                       area=signed_area,
                       out=analysis, cadence=6, force=force)
        path = analysis / recipe["output"]
    else:
        # Reuse the public fetch route, including its inventory checks and
        # endpoint policy. A donor is an analysis, so only lead zero is read.
        tokens = [sys.executable, "-m", "woof", "fetch", "--source", recipe["source"],
                  "--cycle", cycle.strftime("%Y-%m-%dT%H"), "--hours", "0", "--out", str(analysis)]
        if recipe.get("area"):
            tokens += ["--area", signed_area]
        tokens += recipe.get("arguments", [])
        if force:
            tokens.append("--force-refetch")
        subprocess.run(tokens, check=True)
        # Some GRIB analyses publish atmosphere and soil in separate objects.
        # Their ordered record streams are one legal GRIB file; declare the
        # input patterns and output name in the acquisition table.
        if recipe.get("inputs"):
            inputs = []
            for pattern in recipe["inputs"]:
                matches = sorted(analysis.glob(pattern))
                if len(matches) != 1:
                    raise ValueError(f"donor recipe expected one {pattern}, found {len(matches)}")
                inputs += matches
            import shutil
            with (analysis / recipe["output"]).open("wb") as merged:
                for item in inputs:
                    with item.open("rb") as source:
                        shutil.copyfileobj(source, merged, length=8 * 1024 * 1024)
        candidates = list(analysis.glob(recipe["output"]))
        if not candidates and not recipe.get("inputs"):
            # The fetch's own list of the payloads it delivered (inputs.txt), when the table's name is not
            # the one this transport wrote. Named breakage (WOOF run_6b3482bc3f8d1785, 2026-10-26): under
            # WOOF_FETCH_POLICY=aws the GDAS analysis arrives whole from S3 as gdas.tHHz.pgrb2.0p25.f000,
            # not as the NOMADS subset (gdas.*.subset.grib2) the table names, so every Earth-2 run from a
            # recent GFS cycle stopped at 'donor recipe expected one gdas.*.subset.grib2, found 0'.
            from woof import fetch_routes
            listing = analysis / fetch_routes.INPUT_LIST_NAME
            if listing.is_file():
                candidates = [Path(line.strip()) for line in listing.read_text(encoding="utf-8").splitlines()
                              if line.strip() and Path(line.strip()).is_file()]
        if len(candidates) != 1:
            raise ValueError(f"donor recipe expected one {recipe['output']}, found {len(candidates)}")
        path = candidates[0]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {**selected, "analysis": str(path.resolve()),
            "sha256": digest,
            "parallelism": f"{os.environ['GPUWM_RW_ZARR_WORKERS']} threads / network",
            "supplements": {"physical_analysis_surface_data": str(path.resolve())}}


def preparation_profile(args, adapter):
    """Select only a sealed donor variant, from this fetch's own receipt."""
    listing = getattr(args, "input_list", None)
    candidates = []
    if listing is not None:
        listing = Path(listing)
        candidates.append(listing.parent / RECEIPT)
        # Noise-member handoffs bind the shared batch receipt explicitly.
        handoff_path = listing.parent / "prep-arguments.json"
        if handoff_path.is_file():
            handoff = json.loads(handoff_path.read_text())
            if handoff.get("member_identity") == "earth2-noise-seed-v1":
                candidates.insert(0, Path(handoff["parent_receipt"]))
        if listing.is_file():
            candidates += [Path(path).parent / RECEIPT
                           for path in listing.read_text().splitlines() if path.strip()]
    candidates += [Path(path).parent / RECEIPT
                   for path in (getattr(args, "mapped_inputs", None) or ())]
    receipt = next((path for path in candidates if path.is_file()), None)
    if receipt is None:
        return adapter.packaged_profile
    report = json.loads(receipt.read_text())
    if report.get("schema") == "gpuwm-earth2-fetch-v1":
        # Legacy single-trajectory receipts bind only the model's base profile.
        # They have no generated source/donor fields and remain immutable.
        name = report.get("model")
        if not isinstance(name, str) or not name:
            raise ValueError("legacy preparation receipt has no model identity")
        model_id, row = resolve_model(name)
        if adapter.source_id not in ("earth2", f"earth2-{model_id}"):
            raise ValueError("preparation receipt belongs to a different generated source")
        recorded = report.get("model_row", {})
        if not isinstance(recorded, dict) or any(
            recorded.get(key) != row.get(key)
            for key in ("earth2studio_class", "profile", "step_hours")
        ):
            raise ValueError("legacy preparation model declaration differs from the authority")
        if report.get("profile") != row["profile"]:
            raise ValueError("legacy preparation receipt must use its declared base profile")
        return row["profile"]
    model_id, row = resolve_model(report["source"] if adapter.source_id == "earth2"
                                  else adapter.source_id)
    if report["source"] != f"earth2-{model_id}":
        raise ValueError("preparation receipt belongs to a different generated source")
    selected = report["donor"]
    declared = model_table()["donor_policy"]["profile_variants"][row["profile"]]
    if declared.get(selected["id"]) != report["profile"]:
        raise ValueError("preparation donor/profile pair is absent from the source table")
    return report["profile"]


def publish_handoffs(receipt, out):
    """Use only forecast and donor paths recorded by this completed fetch."""
    # The generic acquisition handoff already owns every field needed by go.
    from woof import fetch_routes
    from woof.source_adapters import canonical_source_token
    source = receipt["source"]
    prep_source = canonical_source_token(source)
    for member in receipt["members"]:
        paths = [entry["path"] for entry in receipt["runner"]["files"]
                 if entry["member"] == member and entry["lead_hours"] >= receipt["forecast_start_hour"]]
        listing = out / f"inputs-m{member:04d}.txt"
        listing.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
        tokens = ["--source", prep_source, "--input-list", str(listing),
                  "--author-input-manifest", str(out / f"manifest-m{member:04d}.json")]
        for role, path in sorted(receipt["donor"]["supplements"].items()):
            tokens += ["--supplement", f"{role}={path}"]
        # Every member keeps its own handoff; go consumes one selected
        # trajectory. Fetching a list never merges members into one state.
        target = out if len(receipt["members"]) == 1 else out / f"member-{member:04d}"
        target.mkdir(exist_ok=True)
        fetch_routes.write_prep_arguments(out=target, source=source, prep_source=prep_source,
                                          cycle=datetime.fromisoformat(receipt["cycle"]), tokens=tokens,
                                          member=str(member))


def producer_attributes(mapping):
    """Carry profile-declared field and lead-coordinate units to the writer."""
    document = json.loads(Path(mapping).read_text())
    attributes = document.get("producer_attributes", {})
    lead = document.get("coordinates", {}).get("time", {}).get("lead", {})
    for name in lead.get("selector", {}).get("name", []):
        attributes.setdefault(name, {})["units"] = lead["units"]
    return attributes


def reuse_completed(*, root, out, model_id, row, cycle, hours, crop,
                    ic_source, members, forecast_start_hour):
    """Verify recorded bytes, then publish handoffs without a model runtime.

    The original receipt and payloads stay in the shared root. A different
    output directory receives only a receipt and the requested input lists.
    """
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timezone

    root = Path(root).expanduser().resolve()
    path = root / RECEIPT
    if not path.is_file():
        return None
    started = time.perf_counter()
    saved = json.loads(path.read_text())
    def utc(value):
        value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None
                else value.astimezone(timezone.utc))
    if (saved.get("schema") != "gpuwm-earth2-fetch-v2"
            or saved.get("model") != model_id or saved.get("source") != f"earth2-{model_id}"
            or saved.get("model_row") != row
            or saved.get("model_table_sha256") != hashlib.sha256(TABLE.read_bytes()).hexdigest()
            or saved.get("initial_condition_source") != ic_source
            or utc(saved["cycle"]) != utc(cycle.isoformat())
            or not set(members).issubset(saved.get("members", []))):
        return None
    first = saved["forecast_start_hour"]
    last = first + saved["hours"]
    if first > forecast_start_hour or last < forecast_start_hour + hours:
        return None
    # v2 initially stored the padded crop only in the runner request.
    request_path = root / "earth2-request.json"
    if "area" in saved:
        available = saved["area"]
    elif request_path.is_file():
        available = json.loads(request_path.read_text())["area"]
    else:
        return None  # unknown coverage is not evidence of a matching area
    if available is not None and (crop is None or not (
            available[0] <= crop[0] <= crop[2] <= available[2]
            and available[1] <= crop[1] <= crop[3] <= available[3])):
        return None
    same_root = root == out
    start = first if same_root else forecast_start_hour
    end = last if same_root else forecast_start_hour + hours
    files = [item for item in saved["runner"]["files"]
             if item["member"] in members and start <= item["lead_hours"] <= end]
    expected = set(range(start, end + 1, int(row["step_hours"])))
    for member in members:
        leads = [item["lead_hours"] for item in files if item["member"] == member]
        if set(leads) != expected or len(leads) != len(expected):
            raise ValueError(f"Completed fetch has missing or duplicate frames for member {member}")

    def verify(item):
        frame = Path(item["path"])
        if "bytes" in item and frame.stat().st_size != item["bytes"]:
            raise ValueError(f"Completed fetch size differs: {frame}")
        with frame.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Completed fetch hash differs: {frame}")

    # Donor receipts seal their analysis, which also supplies the supplements.
    donor = saved["donor"]
    analysis = donor.get("analysis")
    if not analysis or not donor.get("sha256"):
        return None  # a donor without a recorded digest cannot be reused
    if any(Path(p).resolve() != Path(analysis).resolve()
           for p in donor["supplements"].values()):
        return None
    with ThreadPoolExecutor(max_workers=min(4, len(files) + 1)) as pool:
        list(pool.map(verify, [*files, {"path": analysis, "sha256": donor["sha256"]}]))
    # Keep the same native member/seed checks as preparation's handoff verifier.
    from netCDF4 import Dataset
    for item in files:
        with _HDF5_LOCK, Dataset(item["path"]) as frame:
            if (int(frame.getncattr("earth2_member")) != item["member"] or
                    int(frame.getncattr("earth2_generation_seed")) != item.get("seed", saved["runner"]["seed"])):
                raise ValueError("Completed fetch native member/seed differs: " + item["path"])
    receipt = dict(saved, members=members, hours=end-start, forecast_start_hour=start,
                   area=available, forecast=[item["path"] for item in files],
                   runner=dict(saved["runner"], members=members, files=files),
                   reuse={"root": str(root), "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "verified_files": len(files), "verified_donor": analysis,
                          "seconds": time.perf_counter()-started,
                          "parallelism": "up to four hash threads; serial HDF5 metadata"})
    out.mkdir(parents=True, exist_ok=True)
    if not same_root:
        (out / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
        (out / "fetch-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    publish_handoffs(receipt, out)
    return receipt


def run_model(*, model: str | None, cycle: datetime, hours: int, area, out: Path,
              ic_source: str | None = None, force: bool = False, member=None,
              forecast_start_hour: int = 0, cadence=None, prepare_only: bool = False,
              reuse_fetch=None) -> dict:
    """Generate forcing, or run its whole CPU stage before a card is admitted.

    prepare_only touches no GPU and replaces no forecast file. It fetches and
    audits the weights, stages declared loader inputs, constructs the model on
    the CPU into the constructed-model cache, fetches the initial conditions
    into this run's data cache and acquires the donor. The generating run then
    reuses all of it, so its card lease covers the inference, not the ~16 min
    of CPU model construction (FCN3 on G12, 2026-10-20: 970 s of a 1025 s
    lease). The inference VRAM check stays on the generating path only.
    """
    from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu
    model_id, row = resolve_model(model)
    table = model_table()
    runtime = table["runtime"]
    ic_source = ic_source or row.get("initial_condition_source", "era5")
    step = int(row["step_hours"])
    if hours <= 0 or hours % step or forecast_start_hour < 0 or forecast_start_hour % step:
        raise ValueError(f"hours and forecast_start_hour must follow the row's {step} h clock")
    if cadence is not None and int(cadence) != step:
        raise ValueError(f"this model produces every {step} h; cadence={cadence} would misstate its output")
    members = members_from_flag(member)
    population = row.get("members", {}).get("population")
    if population is not None and max(members) >= int(population):
        raise ValueError(f"member ID must be below the row population {population}")
    if len(members) > 1 and not row.get("stochastic", row.get("ensemble", row.get("members", {}).get("kind") in ("native_noise", "native_ensemble", "perturbation"))):
        raise ValueError("this row declares no stochastic ensemble; a member list would repeat one trajectory")
    ic = runtime["initial_condition_sources"].get(ic_source)
    if ic is None:
        raise ValueError(f"initial-condition source {ic_source!r} is absent from the runtime table")
    out = Path(out).resolve()
    crop = None
    if area is not None:
        south, west, north, east = (float(value) for value in area)
        crop = [max(-90., south - .5), (west - .5) % 360,
                min(90., north + .5), (east + .5) % 360]
        if crop[1] > crop[3]:
            raise ValueError("the selected crop crosses 0 degrees; split it into two areas")
    # Reuse a sealed local or external fetch before inference admission;
    # an already complete forecast needs no GPU or CPU model construction.
    if not force:
        for root in dict.fromkeys([out, *([] if reuse_fetch is None else [Path(reuse_fetch)])]):
            reused = reuse_completed(root=root, out=out, model_id=model_id, row=row,
                                     cycle=cycle, hours=hours, crop=crop, ic_source=ic_source,
                                     members=members, forecast_start_hour=forecast_start_hour)
            if reused is not None:
                return reused
    if not prepare_only and no_local_gpu():
        raise ValueError(f"{NO_LOCAL_GPU_ENV} is set; Earth2 generation cannot open the local GPU. "
                         "Run generation in an admitted GPU environment with this switch unset")
    memory_admission = None if prepare_only else require_inference_vram(model_id, row)
    python = earth2_python()
    out.mkdir(parents=True, exist_ok=True)
    forecast = out / f"earth2-{model_id}-{cycle:%Y%m%dT%H}.nc"
    existing = list(out.glob(forecast.stem + "-m*-f*.nc"))
    if existing and not force and not prepare_only:
        raise ValueError(f"{out} already holds forecast frames; use another --out or --force-refetch")
    if force and not prepare_only:
        for path in existing:
            path.unlink()
    cache_policy = runtime["weight_cache"]
    cache = Path(os.environ.get(cache_policy["environment"], cache_policy["default"])).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    request = {"memory_admission": memory_admission,
               "model_class": row["earth2studio_class"], "ic_class": ic["earth2studio_class"],
               "init": cycle.isoformat(), "steps": (hours + forecast_start_hour) // step,
               "output": str(forecast), "area": crop, "members": members,
               "seed": int(row.get("seed", row.get("member", {}).get("seed", 0))) + members[0], "weight_cache": str(cache),
               "pinned_version": runtime["pinned_version"], "data_cache": str(out / "runtime-cache"),
               "loader_kwargs": row.get("load_model_kwargs", {}) | row.get("loader_kwargs", {}),
               "model_package": row.get("weights", {}).get("uri"),
               "expected_weights": row.get("weights", {}),
               "loader_preparations": row.get("loader_preparations", []),
               "model_calls": row.get("model_calls", []),
               "output_grid": row.get("output_grid"),
               "ic_kwargs": dict(ic.get("kwargs", {}), **row.get("ic_kwargs", {})),
               "runtime_env": row.get("runtime_env", {})}
    if row.get("members"):
        policy = row["members"]
        request["member_policy"] = policy
        request["seeds"] = [int(policy["first_seed"]) + member for member in members]
        request["seed"] = request["seeds"][0]
        request["batch_size"] = policy.get("batch_size", 1)
    from woof.source_authorities import packaged_authorities
    mapping = packaged_authorities(row["profile"])["mapping"]
    request["output_attributes"] = producer_attributes(mapping)
    if (row.get("weights") or {}).get("host") == "huggingface":
        from woof.hf_fetch import fetch as fetch_weights
        request["package_root"] = fetch_weights(model_id)["package_root"]
    request_path = out / ("earth2-loader-request.json" if prepare_only else "earth2-request.json")
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    log = out / ("earth2-loader-preparation.log" if prepare_only else "earth2-runner.log")
    operation = "staging the CPU stage (no GPU) of" if prepare_only else "generating"
    print(f"fetch: {operation} {model_id} members {members}; log {log}", flush=True)
    runner = Path(__file__).with_name("earth2_runner.py")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([python, str(runner),
                                   *(["--prepare-loader-inputs"] if prepare_only else []),
                                   str(request_path)], stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Earth2 runtime exited {completed.returncode}; see {log}")
    report = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    if prepare_only:
        donor_start = time.perf_counter()
        donor = _donor(table=table, row=row, cycle=cycle, area=crop, out=out, force=force)
        report["donor"] = donor
        report["stages"] = report.get("stages", []) + [
            {"stage": "donor", "seconds": round(time.perf_counter() - donor_start, 3),
             "parallelism": donor.get("parallelism", "native threads / network; see donor receipt")}]
        report["wall_seconds"] = round(time.perf_counter() - started, 3)
        (out / "earth2-loader-preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    donor_start = time.perf_counter()
    donor = _donor(table=table, row=row, cycle=cycle, area=crop, out=out, force=force)
    receipt = {"schema": "gpuwm-earth2-fetch-v2", "source": f"earth2-{model_id}",
               "model": model_id, "model_row": row,
               "model_table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
               "initial_condition_source": ic_source, "cycle": cycle.isoformat(),
               "hours": hours, "forecast_start_hour": forecast_start_hour, "area": crop,
               "members": members, "forecast": [item["path"] for item in report["files"]],
               "analysis": donor.get("analysis"), "donor": donor, "profile": donor["profile"],
               "runner": report, "stages": report["stages"] + [
                   {"stage": "donor", "seconds": round(time.perf_counter() - donor_start, 3),
                    "parallelism": donor.get("parallelism", "native threads / network; see donor receipt")}],
               "wall_seconds": round(time.perf_counter() - started, 3)}
    (out / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    publish_handoffs(receipt, out)
    (out / "fetch-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def fetch(args) -> int:
    from woof.fetch import _resolve_area, parse_cycle
    if args.cycle in (None, "latest") or args.hours is None or args.out is None:
        raise ValueError("Earth2 generation needs an explicit --cycle, --hours and --out")
    area = _resolve_area(args)
    if area is not None:
        area = (area.lat_south, area.lon_west, area.lat_north, area.lon_east)
    from woof.source_adapters import get_source_adapter
    if str(args.source).startswith("earth2-") and getattr(args, "model", None):
        if resolve_model(args.source)[0] != resolve_model(args.model)[0]:
            raise ValueError("--source and --model select different model rows")
    if getattr(args, "earth2_members", None) or str(getattr(args, "member", "") or "").startswith("n"):
        from woof.earth2_members import fetch_member
        fetch_member(args, model=getattr(args, "model", None) or get_source_adapter(args.source).model_key,
                     cycle=parse_cycle(args.cycle, "era5"), area=area)
        return 0
    source = str(args.source)
    model = getattr(args, "model", None) or get_source_adapter(source).model_key
    # With no card in reach (CUDA hidden, or the local-device opt-out) the
    # door runs the CPU stage instead of refusing, so it can run outside a
    # card lease; the same command under a lease then generates from it.
    from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    hidden = visible is not None and visible.strip() in ("", "-1")
    prepare_only = bool(getattr(args, "prepare_only", False)) or hidden or no_local_gpu()
    report = run_model(model=model, cycle=parse_cycle(args.cycle, "era5"), hours=int(args.hours), area=area,
                       out=Path(args.out), force=bool(args.force_refetch), member=getattr(args, "member", None),
                       forecast_start_hour=getattr(args, "forecast_start_hour", None) or 0,
                       cadence=getattr(args, "cadence", None), prepare_only=prepare_only,
                       reuse_fetch=getattr(args, "reuse_fetch", None))
    if prepare_only:
        why = ("--prepare-only" if getattr(args, "prepare_only", False) else
               "CUDA devices are hidden" if hidden else f"{NO_LOCAL_GPU_ENV} is set")
        stages = ", ".join(f"{item['stage']} {item['seconds']:g} s" for item in report.get("stages", []))
        print(f"fetch: {source}: CPU stage done in {report['wall_seconds']:g} s with no GPU ({why}): {stages}. "
              f"Receipt {Path(args.out).resolve() / 'earth2-loader-preparation.json'}. "
              "No forecast was generated; run the same command with a card admitted to generate it "
              "from this staged state", flush=True)
    return 0


def prepare_launch(args):
    """Create one ordinary regional CONFIG for the chosen generated member."""
    import argparse
    import shutil
    import tomllib
    from woof import domain_wizard
    from woof.toml_document import emit_experiment_toml
    config = getattr(args, "config", None)
    if config is None:
        if not all(getattr(args, key, None) for key in ("source", "init", "center")):
            raise ValueError("generation needs --source, --init and --center, or CONFIG")
        model_id, _ = resolve_model(args.source)
        values = members_from_flag(getattr(args, "member", None))
        if len(values) != 1:
            raise ValueError("go launches one trajectory; generate a member list with fetch and use its member handoffs")
        identity = {key: getattr(args, key, None) for key in
                    ("source", "init", "center", "hours", "extent_km", "card", "member", "history_interval", "geog_root")}
        identity = {key: str(value) if isinstance(value, Path) else value
                    for key, value in identity.items()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        root = Path(getattr(args, "outdir", None) or "earth2-forecast").resolve() / "configurations" / key
        root.mkdir(parents=True, exist_ok=True)
        config = root / "forecast.toml"
        parser = argparse.ArgumentParser()
        domain_wizard.register_cli(parser.add_subparsers(dest="command"))
        argv = ["domain", "--point=" + args.center, "--source", "earth2-" + model_id,
                "--cycle", args.init, "--hours", str(args.hours), "--root-dx", "3",
                "--point-extent-km", str(args.extent_km), "--history-interval",
                str(getattr(args, "history_interval", None) or 3600),
                "--nz", "49", "--out", str(config)]
        for flag in ("card", "geog_root"):
            if getattr(args, flag, None):
                argv += ["--" + flag.replace("_", "-"), str(getattr(args, flag))]
        code = domain_wizard.domain_main(parser.parse_args(argv))
        if code:
            raise ValueError(f"domain creation exited {code}")
        args.source = args.init = args.center = None
    elif getattr(args, "source", None) or getattr(args, "init", None) or getattr(args, "center", None):
        raise ValueError("CONFIG already owns its source, cycle and domain; use --cycle to re-time it")
    elif getattr(args, "history_interval", None) is not None:
        raise ValueError("CONFIG owns its history interval; edit the configuration")
    config = Path(config)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    forcing = raw.get("fetch", {})
    declared = forcing.get("source")
    if declared in ("earth2", "earth2studio"):
        key, _ = resolve_model(forcing.get("model"))
        forcing["source"] = "earth2-" + key
        forcing.pop("model", None)
        target = config.with_name(config.stem + ".generated.toml")
        target.write_text(emit_experiment_toml(raw), encoding="utf-8")
        namelist = config.with_name(config.stem + ".namelist.wps")
        if namelist.is_file():
            shutil.copyfile(namelist, target.with_name(target.stem + ".namelist.wps"))
        config = target
    elif forcing.get("model") and resolve_model(declared)[0] != resolve_model(forcing["model"])[0]:
        raise ValueError("CONFIG source and model select different model rows")
    if getattr(args, "member", None) is not None:
        members = members_from_flag(args.member)
        if len(members) != 1:
            raise ValueError("go requires one source member; use fetch for a GPU member batch")
        raw = tomllib.loads(config.read_text(encoding="utf-8"))
        raw["fetch"]["member"] = str(members[0])
        target = config.with_name(config.stem + f".member-{members[0]:04d}.toml")
        target.write_text(emit_experiment_toml(raw), encoding="utf-8")
        namelist = config.with_name(config.stem + ".namelist.wps")
        if namelist.is_file():
            shutil.copyfile(namelist, target.with_name(target.stem + ".namelist.wps"))
        config = target
    args.config = config


def validate_member(source, member):
    if isinstance(member, str) and member.startswith("n"):
        from woof.earth2_members import validate_member as validate_noise
        return [validate_noise(member)]
    members = members_from_flag(member)
    if source != "earth2":
        _, row = resolve_model(source)
        if len(members) > 1 and not row.get("stochastic", row.get("ensemble", row.get("members", {}).get("kind") in ("native_noise", "native_ensemble", "perturbation"))):
            raise ValueError("the selected row declares no stochastic ensemble")
    return members


def verify_handoff(hints, handoff, *, out):
    """Bind generated-member provenance and actual bytes before preparation."""
    if isinstance(hints.get("member"), str) and hints["member"].startswith("n"):
        from woof.earth2_members import verify_handoff as verify_noise
        return verify_noise(hints, handoff, out=out)
    from concurrent.futures import ThreadPoolExecutor
    from woof.fetch import parse_cycle
    from woof.member_grammar import MemberIdentityRefusal
    from woof.source_adapters import get_source_adapter
    source = get_source_adapter(hints["source"]).source_id
    member, = members_from_flag(hints.get("member"))
    if handoff.get("source") != source or handoff.get("member") != str(member):
        raise MemberIdentityRefusal("Generated source/member differs from the selected handoff")
    if parse_cycle(handoff["cycle"], "era5") != parse_cycle(hints["cycle"], "era5"):
        raise MemberIdentityRefusal("Generated forcing cycle differs from the selected cycle")
    arguments = list(handoff["argv"])
    if arguments.count("--input-list") != 1:
        raise MemberIdentityRefusal("Generated forcing needs exactly one bound input-list")
    listing = Path(arguments[arguments.index("--input-list") + 1]).resolve(strict=True)
    fetched = json.loads((listing.parent / RECEIPT).read_text())
    if (fetched["source"] != source or
            parse_cycle(datetime.fromisoformat(str(fetched["cycle"])).strftime("%Y-%m-%dT%H"), "era5")
            != parse_cycle(hints["cycle"], "era5")):
        raise MemberIdentityRefusal("Generated receipt source/cycle differs from the request")
    expected = [item for item in fetched["runner"]["files"]
                if item["member"] == member and item["lead_hours"] >= fetched["forecast_start_hour"]]
    paths = [str(Path(line).resolve(strict=True)) for line in listing.read_text().splitlines() if line.strip()]
    if not paths or paths != [str(Path(item["path"]).resolve()) for item in expected] or len(set(paths)) != len(paths):
        raise MemberIdentityRefusal("Generated input-list differs from the member's recorded trajectory")
    def check(item):
        with Path(item["path"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise MemberIdentityRefusal("Generated forcing hash differs: " + item["path"])
        return {"path": item["path"], "sha256": digest, "member": member,
                "lead_hours": item["lead_hours"]}
    # HDF5 metadata is inspected serially because its C library is not thread
    # safe; the full-file reads and hashes use independent thread workers.
    from netCDF4 import Dataset
    for item in expected:
        with _HDF5_LOCK, Dataset(item["path"]) as frame:
            if (int(frame.getncattr("earth2_member")) != member or
                    int(frame.getncattr("earth2_generation_seed")) != item.get("seed", fetched["runner"]["seed"])):
                raise MemberIdentityRefusal("Generated frame's member attribute differs: " + item["path"])
    workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(check, expected))
    receipt = {"schema": "arwen.forcing-member.v1", "source": source, "member": str(member),
               "cycle": handoff["cycle"], "model_table_sha256": fetched["model_table_sha256"],
               "weights": fetched["runner"]["weights"], "seed": expected[0].get("seed", fetched["runner"]["seed"]),
               "files": files}
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(encoded.encode()).hexdigest()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"member-inputs-{key}.txt"
    text = "".join(path + "\n" for path in paths)
    try:
        with target.open("x") as stream:
            stream.write(text)
    except FileExistsError:
        if target.read_text() != text:
            raise MemberIdentityRefusal("Verified generated-member input-list is corrupt")
    receipt["input_list"] = str(target.resolve())
    (out / f"member-verification-{key}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
