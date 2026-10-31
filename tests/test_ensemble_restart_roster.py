from conftest import requires_netcdf_writer_bridge
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.ensemble import restart_roster
from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from woof.ensemble.packing import CardBudget
from woof.ensemble.production import PreparedEnsembleSession
from woof.ensemble.runtime_context import current_capture


class Collector:
    def __init__(self, root):
        self.root, self.rows = root, []

    def submit(self, **row):
        self.rows.append(row["member_id"])

    def save_resume(self):
        restart_roster.atomic_json(self.root / ".ensemble-resume" / "collector.json",
            {"rows": self.rows, "files": []})

    def restore_resume(self):
        self.rows = json.loads((self.root / ".ensemble-resume" / "collector.json").read_text())["rows"]

    def finish_run(self):
        return {"members": self.rows}

    def require_complete(self):
        assert self.rows == [0, 1, 2]


def prepared(member):
    experiment = SimpleNamespace(run_seconds=60, restart_interval_s=30,
        start_time=datetime(2024, 1, 1), root=SimpleNamespace(run=SimpleNamespace()))
    return SimpleNamespace(experiment=experiment, member=member, source="fixture",
        prepared_head_sha256=f"member-head-{member}", boundary_interval_seconds=3600)


def session(root, collector, roster=None):
    return PreparedEnsembleSession({"members": 3, "base_seed": 17}, output_directory=root,
        input_provider=lambda **kwargs: prepared(kwargs["member_id"]), restart_roster=roster,
        collector=collector, cards=(CardBudget(0, 1000),),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)),
        device_scope=lambda _: nullcontext())


@pytest.fixture
def identity(monkeypatch):
    from woof.core import model
    monkeypatch.setattr(model, "restart_identity_payload", lambda exp: {"dt": 3, "physics": 16})


def test_finished_members_are_retained_and_fresh_session_restores_the_original_roster(tmp_path, identity):
    first = Collector(tmp_path)
    def interrupted(inputs, **options):
        member = current_capture().member_id
        if member == 1:
            raise KeyboardInterrupt("reclaimed at member 1")
        first.submit(member_id=member)
        return {"status": "PASS", "member": member, "state_sha256": f"member-{member}-state"}
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(interrupted, prepared(0))
    document = json.loads((tmp_path / restart_roster.ROSTER).read_text())
    assert [row["status"] for row in document["members"]] == ["completed", "pending", "pending"]
    second, invoked = Collector(tmp_path), []
    def resumed(inputs, **options):
        member = current_capture().member_id
        invoked.append(member)
        second.submit(member_id=member)
        return {"status": "PASS", "member": member, "state_sha256": f"member-{member}-state"}
    report = session(tmp_path, second, tmp_path / restart_roster.ROSTER).run_prepared(resumed, prepared(0))
    assert report["status"] == "PASS"
    assert invoked == [1, 2]
    assert second.rows == [0, 1, 2]
    assert len(report["completed_member_results"]) == 3
    assert report["member_results"][0]["result"]["state_sha256"] == "member-0-state"
    assert report["ensemble_progress"]["members"][0]["elapsed_seconds"] == 60


def test_wrong_member_seed_and_tampered_retained_file_refuse_before_any_runner(tmp_path, identity):
    first = Collector(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()), prepared(0))
    roster = tmp_path / restart_roster.ROSTER
    original = json.loads(roster.read_text())
    changed = json.loads(roster.read_text())
    changed["identities"]["members"][0]["seed"] += 1
    restart_roster.atomic_json(roster, changed)
    invoked = []
    with pytest.raises(ValueError, match="different member identities"):
        session(tmp_path, Collector(tmp_path), roster).run_prepared(lambda *args, **kwargs: invoked.append(1), prepared(0))
    assert not invoked
    restart_roster.atomic_json(roster, original)
    # The sealed copy is the resume authority; the live file may move on.
    sealed = next(item["snapshot"] for item in original["files"] if item["path"] == ".ensemble-resume/collector.json")
    (tmp_path / sealed).write_text("tampered")
    with pytest.raises(ValueError, match="hash-mismatched"):
        session(tmp_path, Collector(tmp_path), roster).run_prepared(lambda *args, **kwargs: invoked.append(1), prepared(0))
    assert not invoked


def test_checkpoint_is_passed_only_to_its_member_and_all_domains_are_hash_verified(tmp_path, identity, monkeypatch):
    from woof.io import restart
    from woof import supervisor
    first, checkpoint = Collector(tmp_path), tmp_path / "members" / "member-0001" / "gpuwmrst_d01_2024-01-01_00_00_30.npz"
    monkeypatch.setattr(restart, "read_restart_header", lambda path: {"elapsed_seconds": 30})
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", lambda path: path)
    def interrupted(inputs, **options):
        if current_capture().member_id == 1:
            checkpoint.write_bytes(b"checkpoint-member-1")
            options["progress_callback"](model_elapsed_seconds=30, outer_step=10, last_checkpoint=checkpoint)
            raise KeyboardInterrupt()
        first.submit(member_id=0)
        return {"status": "PASS"}
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(interrupted, prepared(0))
    roster = tmp_path / restart_roster.ROSTER
    document = json.loads(roster.read_text())
    assert document["members"][1]["status"] == "checkpoint"
    second, paths = Collector(tmp_path), {}
    def resumed(inputs, **options):
        member = current_capture().member_id
        paths[member] = options.get("restart")
        second.submit(member_id=member)
        return {"status": "PASS"}
    session(tmp_path, second, roster).run_prepared(resumed, prepared(0))
    assert paths == {1: checkpoint, 2: None}


def test_scalar_restart_refuses_instead_of_repeating_one_members_state(tmp_path, identity):
    with pytest.raises(ValueError, match="one checkpoint cannot identify every member"):
        session(tmp_path, Collector(tmp_path)).run_prepared(lambda *args, **kwargs: {}, prepared(0), restart=tmp_path / "one.npz")


@pytest.mark.parametrize("sampled_ids", [(2,), (2, 5)])
def test_collector_without_checkpoint_hold_keeps_its_roster_on_one_card(tmp_path, identity, sampled_ids, capsys):
    # A collector that cannot hold member submissions across save and seal
    # would let another card retire packs that collector.json still names.
    collector = Collector(tmp_path)
    run = session(tmp_path, collector)
    run.cards = tuple(CardBudget(device, 1000) for device in sampled_ids)
    run.memory_model = {device: run.memory_model for device in sampled_ids}
    devices = []
    run.device_scope = lambda device: devices.append(device) or nullcontext()
    def runner(inputs, **options):
        collector.submit(member_id=current_capture().member_id)
        return {"status": "PASS"}
    report = run.run_prepared(runner, prepared(0))
    assert report["packing"]["waves"] == 3
    assert set(devices) == {2}
    if len(sampled_ids) > 1:
        assert "members use one card" in capsys.readouterr().err


class HeldCollector(Collector):
    """Production collectors hold submissions while a roster seals."""

    def __init__(self, root):
        super().__init__(root)
        import threading
        self.lock, self.holds = threading.RLock(), 0

    def checkpoint_hold(self):
        self.holds += 1
        return self.lock

    def submit(self, **row):
        with self.lock:
            self.rows.append(row["member_id"])

    def require_complete(self):
        assert sorted(self.rows) == [0, 1, 2]


def test_default_domain_restart_interval_keeps_every_selected_card(tmp_path, identity):
    # `woof domain` writes DEFAULT_RESTART_INTERVAL_S into every config, so
    # a default multi-card ensemble is a checkpointed ensemble.
    from woof.domain_wizard import DEFAULT_RESTART_INTERVAL_S
    assert DEFAULT_RESTART_INTERVAL_S > 0
    collector = HeldCollector(tmp_path)
    run = session(tmp_path, collector)
    run.input_provider = lambda **kwargs: SimpleNamespace(**dict(vars(prepared(kwargs["member_id"])),
        experiment=SimpleNamespace(**dict(vars(prepared(kwargs["member_id"]).experiment),
            run_seconds=7200, restart_interval_s=DEFAULT_RESTART_INTERVAL_S))))
    sampled_ids = (2, 5, 7)
    run.cards = tuple(CardBudget(device, 1000) for device in sampled_ids)
    run.memory_model = {device: run.memory_model for device in sampled_ids}
    devices = []
    run.device_scope = lambda device: devices.append(device) or nullcontext()
    def runner(inputs, **options):
        collector.submit(member_id=current_capture().member_id)
        return {"status": "PASS"}
    report = run.run_prepared(runner, prepared(0))
    assert set(devices) == set(sampled_ids)
    assert report["packing"]["waves"] == 1
    assert collector.holds >= 3
    document = json.loads((tmp_path / restart_roster.ROSTER).read_text())
    assert [row["status"] for row in document["members"]] == ["completed"] * 3


class RetiringCollector(Collector):
    """finish_run retires a private diagnostic its checkpoint snapshot named.

    The real collector does this with .ensemble-diagnostics/<domain>/
    coordinates.nc (retire_private_coordinates), after the last member's
    checkpoint roster was sealed.
    """
    finished = False

    def save_resume(self):
        scratch = self.root / ".ensemble-diagnostics" / "d01" / "coordinates.nc"
        files = []
        if not self.finished:
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.write_bytes(b"worker-only coordinates")
            files.append(scratch.relative_to(self.root).as_posix())
        restart_roster.atomic_json(self.root / ".ensemble-resume" / "collector.json",
            {"rows": self.rows, "files": files})

    def finish_run(self):
        self.finished = True
        (self.root / ".ensemble-diagnostics" / "d01" / "coordinates.nc").unlink(missing_ok=True)
        return super().finish_run()


def test_completed_checkpointed_roster_still_validates(tmp_path, identity):
    # 2.9.0 defect (ensemble showcase, 2026-10-12): a checkpointed roster that
    # completed rc 0 failed its own restart validator ("ensemble resume file is
    # missing or hash-mismatched: .ensemble-diagnostics/d01/coordinates.nc"):
    # the roster was last sealed at the last member's completion, before
    # finish_run retired private diagnostics and the PASS manifest replaced
    # ensemble-run.json.
    collector = RetiringCollector(tmp_path)
    def complete(inputs, **options):
        collector.submit(member_id=current_capture().member_id)
        return {"status": "PASS"}
    report = session(tmp_path, collector).run_prepared(complete, prepared(0))
    assert report["status"] == "PASS"
    identities = json.loads((tmp_path / restart_roster.IDENTITIES).read_text())
    document = restart_roster.verify_roster(tmp_path / restart_roster.ROSTER, identities, root=tmp_path)
    assert [row["status"] for row in document["members"]] == ["completed"] * 3
    assert not any(item["path"].endswith("coordinates.nc") for item in document["files"])


class CancelledCollector(Collector):
    """The failure path cancels the product worker, then seals the roster.

    The real collector's save_resume waits for that worker to go idle and
    raises the cancellation the failure path just caused ("isolated
    ensemble product process exited -15").
    """
    cancelled = False

    def cancel_products(self):
        self.cancelled = True

    def save_resume(self):
        if self.cancelled:
            raise RuntimeError("ensemble product consumer failed: isolated "
                               "ensemble product process exited -15")
        super().save_resume()


def test_member_error_survives_the_roster_seal_after_cancellation(tmp_path, identity):
    # Fresh-install QA (2026-10-12, published 2.8.9): a member failed with
    # "member cumulative precipitation reset", and the user saw only the
    # product worker's -15 exit, because the failure path's roster seal
    # raised the cancellation it had just caused and replaced the error.
    collector = CancelledCollector(tmp_path)
    def failing(inputs, **options):
        member = current_capture().member_id
        if member == 1:
            raise ValueError("member cumulative precipitation reset; QPF would be invalid")
        collector.submit(member_id=member)
        return {"status": "PASS"}
    with pytest.raises(ValueError, match="cumulative precipitation reset") as caught:
        session(tmp_path, collector).run_prepared(failing, prepared(0))
    assert any("exited -15" in note for note in getattr(caught.value, "__notes__", ()))
    manifest = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert manifest["status"] == "failed"
    assert "cumulative precipitation reset" in manifest["error"]
    assert "exited -15" in manifest["restart_roster_seal_error"]


def test_production_collector_hold_blocks_other_member_submissions(tmp_path):
    import threading
    from woof.ensemble.batch_product_output import HeadlineDiagnosticCollector
    collector = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=datetime(2024, 1, 1), array_module=__import__("numpy"))
    acquired = []
    with collector.checkpoint_hold():
        other = threading.Thread(target=lambda: acquired.append(collector._lock.acquire(timeout=0.2)))
        other.start()
        other.join()
    assert acquired == [False]


def test_retained_history_seal_hashes_files_concurrently_without_changing_proofs(tmp_path, monkeypatch):
    import hashlib
    import threading
    from woof.ingest import cpu_backend

    identities = {"schema": restart_roster.CONTRACT, "member_order": [], "members": []}
    restart_roster.atomic_json(tmp_path / restart_roster.IDENTITIES, identities)
    restart_roster.atomic_json(tmp_path / "ensemble-run.json", {"members_completed": []})
    histories = [tmp_path / f"history-{number}.nc" for number in range(2)]
    for number, path in enumerate(histories):
        path.write_bytes(bytes([number]) * 4096)
    monkeypatch.setattr(cpu_backend, "available_cpu_count", lambda: 2)
    original = restart_roster.digest_file
    barrier = threading.Barrier(2)
    threads = set()
    lock = threading.Lock()

    def digest(path):
        if Path(path).name.startswith("history-"):
            with lock:
                threads.add(threading.get_ident())
            # Serial sealing, verification or restoration cannot reach both
            # readers; no timing/speed guess.
            barrier.wait(timeout=5)
        return original(path)

    monkeypatch.setattr(restart_roster, "digest_file", digest)
    record = restart_roster.seal(tmp_path, collector_files=[path.name for path in histories])
    assert len(threads) == 2
    assert [row["path"] for row in record["files"]] == sorted(row["path"] for row in record["files"])
    for row in record["files"]:
        snapshot = tmp_path / row["snapshot"]
        assert row["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
        assert row["bytes"] == snapshot.stat().st_size
    assert restart_roster.verify_roster(tmp_path / restart_roster.ROSTER, identities,
                                      root=tmp_path) == record
    assert restart_roster.restore_roster(record, root=tmp_path)["restored"] == []


def test_sealing_preserves_live_writer_revision_and_reuses_only_sealed_inodes(tmp_path):
    from woof.output_identity import completed_file_record, publication_revision

    identities = {"schema": restart_roster.CONTRACT, "member_order": [], "members": []}
    restart_roster.atomic_json(tmp_path / restart_roster.IDENTITIES, identities)
    restart_roster.atomic_json(tmp_path / "ensemble-run.json", {"members_completed": []})
    history = tmp_path / "wrfout_d01_frame"
    history.write_bytes(b"writer-owned history")
    published = publication_revision(history)
    first = restart_roster.seal(tmp_path, collector_files=[history.name])
    item = next(row for row in first["files"] if row["path"] == history.name)
    snapshot = tmp_path / item["snapshot"]
    sealed_inode = snapshot.stat().st_ino
    assert sealed_inode != history.stat().st_ino
    # A different member can seal after publication and before this writer's
    # completion hash. Linking the live inode made this exact call fail.
    completed_file_record(history, published=published)
    second = restart_roster.seal(tmp_path, collector_files=[history.name])
    item = next(row for row in second["files"] if row["path"] == history.name)
    assert (tmp_path / item["snapshot"]).stat().st_ino == sealed_inode
    assert not snapshot.exists()
    completed_file_record(history, published=published)
    # Later live writes cannot mutate the durable copy.
    history.write_bytes(b"changed live history")
    assert (tmp_path / item["snapshot"]).read_bytes() == b"writer-owned history"


def test_seal_survives_a_reader_hard_linking_the_live_source_during_the_copy(tmp_path, monkeypatch):
    # WoFS run_8dd42875d520e1d3 (2026-10-26): a reader hard-linked a member's
    # live restart files once a second. A link changes ctime, not a byte, and
    # the seal counted ctime, so it repeated its copy and then failed the
    # member with "ensemble snapshot source changed during copy".
    import os
    import time
    identities = {"schema": restart_roster.CONTRACT, "member_order": [], "members": []}
    restart_roster.atomic_json(tmp_path / restart_roster.IDENTITIES, identities)
    restart_roster.atomic_json(tmp_path / "ensemble-run.json", {"members_completed": []})
    live = tmp_path / "gpuwmrst_d02_frame.npz"
    live.write_bytes(b"write-once checkpoint words")
    original, links = restart_roster.digest_file, []
    def digest_while_a_reader_links(path):
        value = original(path)
        if Path(path).name == live.name and Path(path) != live:
            # The reader links on every pass, as the worker did once a second.
            time.sleep(0.05)  # past the filesystem's coarse timestamp tick
            reader = tmp_path / f"reader-link-{len(links)}.npz"
            os.link(live, reader)
            links.append(reader)
        return value
    monkeypatch.setattr(restart_roster, "digest_file", digest_while_a_reader_links)
    document = restart_roster.seal(tmp_path, collector_files=[live.name])
    assert len(links) == 1  # one copy, no repeated pass
    item = next(row for row in document["files"] if row["path"] == live.name)
    assert (tmp_path / item["snapshot"]).read_bytes() == b"write-once checkpoint words"
    # A content change still counts.
    live.write_bytes(b"rewritten checkpoint words, longer")
    assert restart_roster._source_revision(live) != item["source_revision"]


def test_seal_retries_when_another_member_retires_a_checkpoint_mid_pass(tmp_path, identity, monkeypatch):
    from woof.io import restart
    from woof import supervisor
    collector = Collector(tmp_path)
    member = tmp_path / "members" / "member-0001"
    old = member / "gpuwmrst_d01_2024-01-01_00_00_30.npz"
    new = member / "gpuwmrst_d01_2024-01-01_00_00_40.npz"
    monkeypatch.setattr(restart, "read_restart_header",
        lambda path: {"elapsed_seconds": 40 if "00_40" in Path(path).name else 30})
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", lambda path: path)
    def interrupted(inputs, **options):
        if current_capture().member_id == 1:
            old.write_bytes(b"old")
            options["progress_callback"](model_elapsed_seconds=30, outer_step=10, last_checkpoint=old)
            new.write_bytes(b"new")
            options["progress_callback"](model_elapsed_seconds=40, outer_step=12, last_checkpoint=new)
            raise KeyboardInterrupt()
        collector.submit(member_id=0)
        return {"status": "PASS"}
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, collector).run_prepared(interrupted, prepared(0))
    # Another card's runner retires the older set while this pass hashes it.
    original, retired = restart_roster.digest_file, []
    def digest_then_retire(path):
        value = original(path)
        if Path(path) == new and not retired:
            new.unlink()
            retired.append(path)
        return value
    monkeypatch.setattr(restart_roster, "digest_file", digest_then_retire)
    document = restart_roster.seal(tmp_path, collector=collector)
    assert retired
    row = document["members"][1]
    assert row["status"] == "checkpoint" and row["checkpoint"].endswith("00_00_30.npz")
    assert all((tmp_path / item["path"]).is_file() for item in document["files"])


def test_seal_links_write_once_checkpoint_sets_instead_of_copying_them(tmp_path, identity, monkeypatch):
    # U-E10: every roster seal copied each member's write-once checkpoint set
    # into its generation, 44-59 GB of host disk per sealed set on a
    # checkpointed WoFS ensemble. A committed set is published by os.replace
    # and never rewritten, so the sealed copy is a hard link: no added bytes.
    from woof.io import restart
    from woof import supervisor
    collector = Collector(tmp_path)
    member = tmp_path / "members" / "member-0001"
    first = member / "gpuwmrst_d01_2024-01-01_00_00_30.npz"
    later = member / "gpuwmrst_d01_2024-01-01_00_00_40.npz"
    monkeypatch.setattr(restart, "read_restart_header",
        lambda path: {"elapsed_seconds": 40 if "00_40" in Path(path).name else 30})
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", lambda path: path)
    copied = []
    real_copy = restart_roster.shutil.copyfile
    monkeypatch.setattr(restart_roster.shutil, "copyfile",
        lambda source, destination: copied.append(Path(source).name) or real_copy(source, destination))
    def publish(path, words):
        pending = path.with_name(path.name + ".tmp")
        pending.write_bytes(words)
        pending.replace(path)  # as restart._publish_restart_payload does
    def interrupted(inputs, **options):
        if current_capture().member_id == 1:
            member.mkdir(parents=True, exist_ok=True)
            publish(first, b"checkpoint 30 words")
            options["progress_callback"](model_elapsed_seconds=30, outer_step=10, last_checkpoint=first)
            publish(later, b"checkpoint 40 words")
            options["progress_callback"](model_elapsed_seconds=40, outer_step=12, last_checkpoint=later)
            raise KeyboardInterrupt()
        collector.submit(member_id=0)
        return {"status": "PASS"}
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, collector).run_prepared(interrupted, prepared(0))
    document = json.loads((tmp_path / restart_roster.ROSTER).read_text())
    assert document["members"][1]["checkpoint"].endswith("00_00_40.npz")
    item = next(row for row in document["files"] if row["path"].endswith("00_00_40.npz"))
    snapshot = tmp_path / item["snapshot"]
    assert snapshot.stat().st_ino == later.stat().st_ino  # one inode, no second copy
    assert not [name for name in copied if name.endswith(".npz")]
    # Retiring the live set leaves the sealed one, and the roster still verifies.
    later.unlink()
    assert snapshot.read_bytes() == b"checkpoint 40 words"
    identities = json.loads((tmp_path / restart_roster.IDENTITIES).read_text())
    assert restart_roster.verify_roster(tmp_path / restart_roster.ROSTER, identities, root=tmp_path)
    assert restart_roster.restore_roster(document, root=tmp_path)["restored"] == [item["path"]]
    assert later.read_bytes() == b"checkpoint 40 words"


@requires_netcdf_writer_bridge
def test_real_rust_collector_snapshot_retains_endpoint_words_and_skips_delivered_output(tmp_path):
    import hashlib
    import numpy as np
    from woof.ensemble.batch_products import FieldProducts
    from woof.ensemble.batch_product_output import HeadlineDiagnosticCollector, NativeDiagnosticSpool
    start = datetime(2024, 1, 1)
    request = FieldProducts("wind10", "m s-1", (25.,))
    coords = np.zeros((2, 2), np.float32)
    geometry = hashlib.sha256(coords.tobytes() + coords.tobytes()).hexdigest()
    original = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=start, requests=(request,), array_module=np)
    spool = NativeDiagnosticSpool(tmp_path, members=2, requests=(request,),
        latitude=coords, longitude=coords, renderer="rw_wrfbatch")
    pack = tmp_path / ".ensemble-diagnostics" / "retained-pack.nc"
    pack.parent.mkdir()
    pack.write_bytes(b"retained member diagnostics")
    valid = "2024-01-01_00:00:00"
    spool.frames[valid] = {"valid_time": valid, "domain": "d01", "status": "pending",
        "members_received": [0], "members_expected": 2, "packs": [{"path": pack, "member_ids": (0,),
            "sha256": hashlib.sha256(pack.read_bytes()).hexdigest(), "bytes": pack.stat().st_size}],
        "products": [], "maps": [], "available_fields": ["wind10"], "unavailable_fields": []}
    spool._manifest()
    original.spools[("d01", geometry)] = spool
    original.cohorts[(1, 0, valid)] = {geometry: {"members": {0}, "spool": spool}}
    key = (1, 0, 0, geometry)
    words = np.array([0, 0x80000000, 0x00000001, 0x3f800001], np.uint32).view(np.float32).reshape(2, 2)
    original.rain_history[key] = {"initial": words.copy(), 0: words.copy()}
    original.rain_initial_ticks[key] = 0
    original.output_ticks.add((key, 0))
    document = original.save_resume()
    assert pack.relative_to(tmp_path).as_posix() in document["files"]
    restored = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=start, requests=(request,), array_module=np, resume=True)
    restored.restore_resume()
    assert restored.rain_history[key]["initial"].tobytes() == words.tobytes()
    assert restored.rain_history[key][0].tobytes() == words.tobytes()
    assert restored.spools[("d01", geometry)].frames[valid]["members_received"] == [0]
    assert restored.spools[("d01", geometry)].frames[valid]["packs"][0]["path"] == pack
    assert restored.submit(state=SimpleNamespace(physics=None), streamed=None,
        metadata={"XLAT": coords, "XLONG": coords}, refl_field=None,
        valid_time=start, grid_id=1, episode=0, member_id=0) is None
    assert restored._kernels == {}
    assert pack.read_bytes() == b"retained member diagnostics"


class FrameWritingCollector(Collector):
    """Names the files the real collector rewrites on every frame."""

    def write_frame(self, frame):
        manifest = self.root / "d01" / "ensemble-manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        restart_roster.atomic_json(manifest, {"frames": frame})
        restart_roster.atomic_json(self.root / "ensemble-history-ledger.json", {"records": frame})

    def save_resume(self):
        packs = sorted(path.relative_to(self.root).as_posix()
                       for path in (self.root / ".ensemble-diagnostics").rglob("*.nc"))
        restart_roster.atomic_json(self.root / ".ensemble-resume" / "collector.json",
            {"rows": self.rows, "files": ["d01/ensemble-manifest.json", "ensemble-history-ledger.json", *packs]})

    def restore_resume(self):
        super().restore_resume()
        self.restored_frames = json.loads((self.root / "d01" / "ensemble-manifest.json").read_text())["frames"]


def test_killed_ensemble_resumes_after_frames_rewrote_the_sealed_files(tmp_path, identity, monkeypatch):
    # 2.9.1 audit (woof 2.9.0 from PyPI): an ensemble killed after member 1's
    # checkpoint seal refused every resume with "ensemble resume file is missing or
    # hash-mismatched: d01/ensemble-manifest.json". Member 1's next frame rewrote
    # the domain manifest and the history ledger 0.5 s after the seal, and the
    # roster named only those live files. A SIGKILL leaves no later seal.
    from woof.io import restart
    from woof import supervisor
    first = FrameWritingCollector(tmp_path)
    first.write_frame(["13:00"])
    scratch = tmp_path / ".ensemble-diagnostics" / "d01" / "frame-14"
    scratch.mkdir(parents=True)
    (scratch / "pack-000000.nc").write_bytes(b"member 0 pack, sealed")
    member = tmp_path / "members" / "member-0001"
    sealed = member / "gpuwmrst_d01_2024-01-01_00_00_30.npz"
    unsealed = member / "gpuwmrst_d01_2024-01-01_00_00_40.npz"
    monkeypatch.setattr(restart, "read_restart_header",
        lambda path: {"elapsed_seconds": 40 if "00_40" in Path(path).name else 30})
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", lambda path: path)
    def killed(inputs, **options):
        if current_capture().member_id == 1:
            sealed.write_bytes(b"checkpoint 30")
            options["progress_callback"](model_elapsed_seconds=30, outer_step=10, last_checkpoint=sealed)
            # The process dies before any later seal (SIGKILL, node loss).
            monkeypatch.setattr(restart_roster, "seal", lambda *args, **kwargs: None)
            first.write_frame(["13:00", "14:00"])
            # The next frame's pack and render cache; their writers refuse to
            # replace an unowned file ("diagnostic scratch path already exists").
            (scratch / "pack-000001.nc").write_bytes(b"member 1 pack, after the seal")
            (scratch / "aggregate-render-source.nc").write_bytes(b"after the seal")
            unsealed.write_bytes(b"checkpoint 40, never sealed")
            sealed.unlink()
            raise KeyboardInterrupt()
        first.submit(member_id=0)
        return {"status": "PASS"}
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(killed, prepared(0))
    monkeypatch.undo()
    from woof.core import model
    monkeypatch.setattr(model, "restart_identity_payload", lambda exp: {"dt": 3, "physics": 16})
    monkeypatch.setattr(restart, "read_restart_header", lambda path: {"elapsed_seconds": 30})
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", lambda path: path)
    roster = tmp_path / restart_roster.ROSTER
    assert [row["status"] for row in json.loads(roster.read_text())["members"]] == ["completed", "checkpoint", "pending"]
    second, paths = FrameWritingCollector(tmp_path), {}
    def resumed(inputs, **options):
        member_id = current_capture().member_id
        paths[member_id] = options.get("restart")
        second.submit(member_id=member_id)
        return {"status": "PASS"}
    report = session(tmp_path, second, roster).run_prepared(resumed, prepared(0))
    assert report["status"] == "PASS"
    assert second.restored_frames == ["13:00"]
    assert paths == {1: sealed, 2: None}
    assert sealed.read_bytes() == b"checkpoint 30"
    assert not unsealed.exists()
    assert sorted(path.name for path in scratch.iterdir()) == ["pack-000000.nc"]


def test_tampered_sealed_copy_refuses_and_names_the_remedy(tmp_path, identity):
    first = FrameWritingCollector(tmp_path)
    first.write_frame(["13:00"])
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()), prepared(0))
    roster = tmp_path / restart_roster.ROSTER
    item = next(row for row in json.loads(roster.read_text())["files"] if row["path"] == "d01/ensemble-manifest.json")
    (tmp_path / item["snapshot"]).write_text("tampered")
    invoked = []
    with pytest.raises(ValueError, match="hash-mismatched: d01/ensemble-manifest.json .*new --outdir"):
        session(tmp_path, FrameWritingCollector(tmp_path), roster).run_prepared(
            lambda *args, **kwargs: invoked.append(1), prepared(0))
    assert not invoked


def test_checkpointed_members_share_card_without_losing_durable_roster(tmp_path, identity, monkeypatch):
    from contextlib import contextmanager
    from threading import Barrier
    from woof.ensemble import member_stream
    barrier = Barrier(2)
    @contextmanager
    def member_scope(**kwargs):
        yield SimpleNamespace(receipt=lambda: {"member_id": kwargs["member_id"], "device_id": kwargs["device_id"]})
    monkeypatch.setattr(member_stream, "member_cuda_scope", member_scope)
    collector = HeldCollector(tmp_path)
    run = session(tmp_path, collector)
    run.array_module = SimpleNamespace(cuda=SimpleNamespace(Stream=lambda: None,
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None))
    def runner(inputs, **options):
        member = current_capture().member_id
        if member in (0, 1):
            barrier.wait(timeout=5)
        collector.submit(member_id=member)
        return {"status": "PASS"}
    report = run.run_prepared(runner, prepared(0))
    assert report["packing"]["batches"][0]["member_indices"] == [0, 1]
    assert sorted(report["members_completed"]) == [0, 1, 2]
    roster = json.loads((tmp_path / restart_roster.ROSTER).read_text())
    assert [row["status"] for row in roster["members"]] == ["completed"] * 3


# The collector opens the Rust NetCDF writer when it is built; a native-free
# run (the cut's science stage) skips it, as it does the real-collector case above.
@requires_netcdf_writer_bridge
@pytest.mark.parametrize("return_tick", [1, 2])
def test_moving_nest_resume_keeps_current_rain_placement(tmp_path, return_tick):
    import numpy as np
    from woof.ensemble.batch_products import FieldProducts
    from woof.ensemble.batch_product_output import HeadlineDiagnosticCollector
    options = dict(members=1, renderer="rw_wrfbatch", start_time=datetime(2024, 1, 1),
        requests=(FieldProducts("rain_total", "mm", (1,)),), array_module=np)
    original = HeadlineDiagnosticCollector(tmp_path, **options)
    # Return to the first footprint after a move: its new counters start over.
    for tick, geometry, rain in ((0, "a", 10), (1, "b", 5), (return_tick, "a", 0)):
        key = original._placement_key(2, 0, 0, geometry, advance=True)
        original._record_rain(key, tick, np.full((2, 2), rain, np.float32))
    original.save_resume()
    restored = HeadlineDiagnosticCollector(tmp_path, resume=True, **options)
    restored.restore_resume()
    resumed_key = restored._placement_key(2, 0, 0, "a", advance=True)
    assert resumed_key == key
    restored._record_rain(resumed_key, 3, np.ones((2, 2), np.float32))
    next_key = restored._placement_key(2, 0, 0, "b", advance=True)
    assert next_key[-1] == 3
    restored._record_rain(next_key, 4, np.zeros((2, 2), np.float32))


def test_resume_reuses_only_exact_prepared_bindings(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from woof import stage_cli, prepared_single_domain_forecast
    from woof.ensemble import door, recipe_door

    rows = [dict(member_id=member, prepared_root=str(tmp_path / bank),
                 experiment_config=str(tmp_path / config), wps_namelist=wps)
            for member, bank, config, wps in (
                (0, "bank-a", "a.toml", None),
                (1, "bank-a", "a.toml", None),
                (2, "bank-b", "b.toml", "b.wps"),
                (3, "bank-b", "b.toml", "b.wps"),
                (4, "bank-b", "different.toml", "b.wps"),
                (5, "bank-b", "b.toml", "different.wps"))]
    restart_roster.atomic_json(tmp_path / restart_roster.INPUTS, {"members": rows})
    restart_roster.atomic_json(tmp_path / restart_roster.IDENTITIES,
                              {"member_order": list(range(len(rows)))})
    monkeypatch.setattr(restart_roster, "verify_roster", lambda *a, **kw: None)
    monkeypatch.setattr(stage_cli, "resolve_bundle", lambda root: {"layout": "single"})
    monkeypatch.setattr(stage_cli, "sim_command", lambda *a, **kw: ["python", "-m", "runner"])
    opened, captured = [], {}

    def load(shared, row):
        opened.append(row["member_id"])
        return object()

    @contextmanager
    def scope(*a, **kw):
        captured.update(kw)
        yield

    def run(*a, **kw):
        shared = object()
        provider = captured["input_provider"]
        # A later member can request the first bank before member zero.
        values = {m: provider(shared_inputs=shared, member_id=m)
                  for m in (1, 0, 2, 3, 4, 5)}
        assert values[0] is values[1] is shared
        assert values[2] is values[3]
        assert len({values[m] for m in (0, 2, 4, 5)}) == 4
        for m in range(len(rows)):
            assert provider(shared_inputs=shared, member_id=m) is values[m]
        return 0

    monkeypatch.setattr(recipe_door, "member_inputs", load)
    monkeypatch.setattr(door, "production_run_scope", scope)
    monkeypatch.setattr(prepared_single_domain_forecast, "main", run)
    args = SimpleNamespace(restart_roster=tmp_path / restart_roster.ROSTER, outdir=tmp_path)
    assert restart_roster.resume_prepared_roster(args, object()) == 0
    assert opened == [2, 4, 5]


def test_member_completion_seal_defers_when_the_product_drain_is_busy(tmp_path, identity):
    # run_4a75cc2fe19d92f7 (WOOF, 2026-10-28): both members of a 6 h HRRR
    # WoFS run finished, but the roster seal at member completion hit the
    # 60 s bounded product drain (TimeoutError from save_resume) and failed
    # the run. It now defers like CheckpointProgress: the previous roster
    # stays durable and the end-of-run seal, after the unbounded drain in
    # require_complete, seals the completed tree.
    class BusyDrain(Collector):
        drained = False

        def save_resume(self):
            if self.rows and not self.drained:
                raise TimeoutError("ensemble checkpoint product drain exceeded its bounded hold")
            super().save_resume()

        def require_complete(self):
            super().require_complete()
            self.drained = True

    owner = BusyDrain(tmp_path)
    def runner(inputs, **options):
        member = current_capture().member_id
        owner.submit(member_id=member)
        return {"status": "PASS", "member": member}
    report = session(tmp_path, owner).run_prepared(runner, prepared(0))
    assert report["status"] == "PASS"
    deferred = report["ensemble_checkpoint_deferrals"]
    assert sorted(member for row in deferred for member in row["member_ids"]) == [0, 1, 2]
    assert {row["reason"] for row in deferred} == {"product drain exceeded bounded checkpoint hold"}
    assert all(row["result"]["ensemble_checkpoint_deferred"] == "product drain exceeded bounded checkpoint hold"
               for row in report["member_results"])
    roster = json.loads((tmp_path / restart_roster.ROSTER).read_text())
    assert [row["status"] for row in roster["members"]] == ["completed"] * 3
    assert restart_roster.verify_roster(tmp_path / restart_roster.ROSTER,
        json.loads((tmp_path / restart_roster.IDENTITIES).read_text()), root=tmp_path)
