"""Native plot scheduling and transfer protocol; fixtures are not weather."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import remote_native_plots as plots, remote_processed_v2 as viewer, remote_preparation_v2 as preparation
from woof import remote_artifacts as ra, remote_cli
from test_remote_artifacts import case, encoded
from test_remote_processed_v2 import NODE_DEFAULT_PRODUCTS, native, query


@pytest.fixture
def prepared(native, monkeypatch):
    c = native.case
    viewer.catalog(query(c, products=[]), c.tmp_path)
    viewer._work_job(c.tmp_path, c.record["id"])
    monkeypatch.setattr(plots, "ensure", lambda *_: None)
    monkeypatch.setattr(preparation, "ensure", lambda *_: None)
    monkeypatch.setattr(plots, "_spacing", lambda *_: 3000.)
    renders = []
    # The compact processor stub the `native` fixture installed stays reachable:
    # a plot pass must never run a process request itself, and the count of the
    # conversions it did not run is what proves it.
    process = plots.subprocess.run
    def render(command, **kwargs):
        if "--process-request" in command:
            return process(command, **kwargs)
        request = json.loads(Path(command[command.index("--render-store-request") + 1]).read_text())
        renders.append(request)
        native_result = json.loads(Path(request["process_result"]).read_text())
        root = Path(request["out_dir"])
        root.mkdir()
        panels = []
        for slug in request["products"]:
            path = root / (slug + ".png")
            path.write_bytes(b"PNG transfer protocol fixture; no meteorology")
            panels.append({"slug": slug, "path": str(path), "sha256": ra._file_sha(path), "bytes": path.stat().st_size})
        result = {"schema": "arwen.native-store-render-result.v1", "frame_id": native_result["frame"]["id"],
            "identity": native_result["frame"]["identity"], "source_store_sha256": native_result["frame"]["rws_sha256"],
            "wrf_imported": False, "volume_store_created": False, "panels": panels}
        Path(command[-1]).write_bytes(encoded(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(plots.subprocess, "run", render)
    return SimpleNamespace(case=c, renders=renders, native=native)


def test_incremental_plots_reuse_compact_store_and_do_not_retry_published_frames(prepared):
    c = prepared.case
    assert plots.work_once(c.tmp_path, c.record["id"])["pending"] == 1
    assert len(prepared.renders) == 1 and len(prepared.native.calls) == 1
    assert prepared.renders[0]["products"] == list(NODE_DEFAULT_PRODUCTS)
    assert prepared.renders[0]["width"] == 1200 and prepared.renders[0]["height"] == 900
    c.status["state"] = "completed"
    state = plots.work_once(c.tmp_path, c.record["id"])
    assert state["done"] and state["ready"] == 1 and state["panels"] == 20
    assert len(prepared.renders) == 1 and c.frame.read_bytes() == c.raw


def test_selected_gallery_roundtrip_reuses_bytes_and_rejects_tampered_commit(prepared, monkeypatch):
    c = prepared.case
    plots.work_once(c.tmp_path, c.record["id"])
    monkeypatch.setattr(remote_cli, "_transport", lambda _, request, **kw: {"ok": True, "native_plots": plots.catalog(request, c.tmp_path)})
    transfers = []
    def download(_, request, path, frame):
        output = io.BytesIO()
        plots.stream(request, c.tmp_path, output)
        assert len(output.getvalue()) == frame["size_bytes"]
        path.write_bytes(output.getvalue())
        transfers.append(request)
    monkeypatch.setattr(ra, "_download", download)
    args = SimpleNamespace(workspace=str(c.tmp_path), job=c.record["id"], domain=1, sequence=1, cache_root=c.tmp_path / "local-plots")
    first = plots.sync(args, [], [])
    assert len(transfers) == 20 and first["transferred_bytes"] > 0
    gallery = Path(first["native_plots"]["gallery_path"])
    assert ra._file_sha(gallery) == first["native_plots"]["gallery_sha256"]
    assert gallery.read_text(encoding="utf-8").count("<figure>") == 20
    assert plots.sync(args, [], [])["transferred_bytes"] == 0 and len(transfers) == 20
    changed = dict(transfers[0], expected_commit_sha256="f" * 64)
    with pytest.raises(ValueError, match="authority changed"):
        plots.stream(changed, c.tmp_path, io.BytesIO())
    changed = dict(transfers[0], product="../../private")
    with pytest.raises(ValueError, match="product or checksum"):
        plots.stream(changed, c.tmp_path, io.BytesIO())


def _gallery_of(reply, count):
    """The node's own published reply with its panel inventory replaced by
    `count` synthetic panels whose checksums and sizes reconcile."""
    import hashlib
    panels = []
    for index in range(count):
        content = f"panel {index} transfer fixture; no meteorology".encode()
        panels.append({"slug": f"var:field_{index:03d}", "sha256": hashlib.sha256(content).hexdigest(),
                       "bytes": len(content), "content": content})
    return {**reply, "panels": [{k: v for k, v in row.items() if k != "content"} for row in panels],
            "bytes": sum(row["bytes"] for row in panels)}, {row["slug"]: row["content"] for row in panels}


@pytest.mark.parametrize("count", [viewer.NODE_PRODUCT_LIMIT, viewer.NODE_PRODUCT_LIMIT + 1])
def test_gallery_inventory_is_bounded_by_the_node_product_limit(prepared, monkeypatch, count):
    # The node accepts a selection of up to NODE_PRODUCT_LIMIT named products
    # and renders that many panels; the desktop must accept the whole gallery
    # back, and refuse one past the bound naming both counts and the door.
    c = prepared.case
    plots.work_once(c.tmp_path, c.record["id"])
    contents = {}
    def transport(_, request, **kw):
        reply, rows = _gallery_of(plots.catalog(request, c.tmp_path), count)
        contents.update(rows)
        return {"ok": True, "native_plots": reply}
    monkeypatch.setattr(remote_cli, "_transport", transport)
    monkeypatch.setattr(ra, "_download", lambda _, request, path, frame: path.write_bytes(contents[request["product"]]))
    args = SimpleNamespace(workspace=str(c.tmp_path), job=c.record["id"], domain=1, sequence=1, cache_root=c.tmp_path / "local-plots")
    if count > viewer.NODE_PRODUCT_LIMIT:
        with pytest.raises(ValueError) as refusal:
            plots.sync(args, [], [])
        assert f"names {count} panels" in str(refusal.value) and f"at most {viewer.NODE_PRODUCT_LIMIT}" in str(refusal.value)
        assert "woof remote list-products" in str(refusal.value)
        return
    result = plots.sync(args, [], [])
    assert len(result["native_plots"]["panels"]) == count and result["transferred_bytes"] > 0
    assert Path(result["native_plots"]["gallery_path"]).read_text(encoding="utf-8").count("<figure>") == count


def test_source_mutation_is_refused_and_reported_once(prepared):
    c = prepared.case
    c.frame.write_bytes(c.raw + b"changed")
    plots.work_once(c.tmp_path, c.record["id"])
    c.status["state"] = "completed"
    state = plots.work_once(c.tmp_path, c.record["id"])
    assert state["done"] and state["failed"] == 1 and not prepared.renders


def test_native_plot_cli_requires_exact_sequence_and_cache():
    from woof.cli import build_parser
    args = build_parser().parse_args(["remote", "sync-native-plots", "--host", "test-host-1", "--python", "/python",
        "--workspace", "/work", "--job", "job-1", "--domain", "2", "--sequence", "3", "--cache-root", "cache"])
    assert args.remote_action == "sync-native-plots" and args.sequence == 3 and args.domain == 2


def test_the_gallery_draws_the_product_set_this_run_selected(prepared):
    """C-296: a job's gallery is the run's own render selection, not a fixed list."""
    c = prepared.case
    c.record["products"] = "2m_temperature,sbcape"
    viewer.catalog(query(c, products=["2m_temperature", "sbcape"]), c.tmp_path)
    viewer._work_job(c.tmp_path, c.record["id"])
    plots.work_once(c.tmp_path, c.record["id"])
    assert prepared.renders[-1]["products"] == ["2m_temperature", "sbcape"]
    assert plots.render_selection(c.record)["products"] == ["2m_temperature", "sbcape"]


def test_sectionless_gallery_keeps_run_maps_and_refuses_explicit_sections():
    record = {"products": "2m_temperature,xsec:QCLOUD=0.01,0.1/wa"}
    assert plots.render_selection(record)["products"] == ["2m_temperature"]
    with pytest.raises(ValueError, match="cannot locate the slice"):
        plots.render_selection(record, {"products": ["xsec:wa"]})


def test_a_requested_size_and_product_set_get_their_own_gallery(prepared):
    """C-225: the render options travel with the request and key the gallery."""
    c = prepared.case
    own = plots.render_selection(c.record)
    request = {**c.request, "action": "native-plots", "sequence": 1, "domain": 1,
               "width": 800, "height": 600, "products": ["2m_temperature"]}
    value = plots.catalog(request, c.tmp_path)
    assert value["width"] == 800 and value["height"] == 600
    assert value["render_id"] != own["render_id"]
    assert value["selection_products"] == ["2m_temperature"]
    assert value["map_products"] is True
    # The watcher now renders both galleries, each into its own receipt.
    plots._work_selections(c.tmp_path, c.record["id"])
    viewer._work_job(c.tmp_path, c.record["id"])
    plots._work_selections(c.tmp_path, c.record["id"])
    sized = [row for row in prepared.renders if row["width"] == 800]
    assert sized and sized[-1]["products"] == ["2m_temperature"] and sized[-1]["height"] == 600
    ready = plots.catalog(request, c.tmp_path)
    assert not ready["waiting"] and [row["slug"] for row in ready["panels"]] == ["2m_temperature"]
    # The run's own gallery is untouched by the reader's selection.
    assert plots.catalog({**c.request, "action": "native-plots", "sequence": 1, "domain": 1},
                         c.tmp_path)["render_id"] == own["render_id"]


def test_a_panel_size_outside_the_renderers_range_names_that_range(prepared):
    c = prepared.case
    with pytest.raises(ValueError) as failure:
        plots.render_selection(c.record, {"width": 12})
    message = str(failure.value)
    assert "256" in message and "4096" in message and "refuse the whole frame" in message


def test_the_gallery_door_registers_its_render_options():
    from woof.cli import build_parser
    args = build_parser().parse_args(["remote", "sync-native-plots", "--host", "test-host-1", "--python", "/python",
        "--workspace", "/work", "--job", "job-1", "--sequence", "3", "--cache-root", "cache",
        "--products", "2m_temperature,var:T2", "--width", "800", "--height", "600"])
    assert plots.request_options(args) == {"products": ["2m_temperature", "var:T2"],
                                           "width": 800, "height": 600}


def _recording_environments(monkeypatch, module):
    """Every renderer call's command and the environment it was handed."""
    seen = []
    inner = module.subprocess.run
    def run(command, **kwargs):
        seen.append((list(command), kwargs.get("env")))
        return inner(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", run)
    return seen


def test_a_wheel_install_hands_the_gallery_renderer_its_map_files(prepared, tmp_path, monkeypatch):
    """THE BREAKAGE: the gallery watcher started the renderer with the caller's
    environment, so the renderer of a pip install was handed no map files and
    every gallery picture of a remote job had no coastlines, borders or state
    lines. Measured on the 5070 Ti host from a wheel install: the same
    --render-store-request call with and without renderer_env differed by
    21,141 px, all of them map lines."""
    from test_render_basemap_delivery import wheel_with_companion
    companion = wheel_with_companion(tmp_path, monkeypatch)
    seen = _recording_environments(monkeypatch, plots)
    c = prepared.case
    state = plots.work_once(c.tmp_path, c.record["id"])
    drawn = [env for command, env in seen if "--render-store-request" in command]
    assert len(drawn) == 1 and prepared.renders
    assert drawn[0] is not None, "the gallery renderer got the caller's environment and no map files"
    assert drawn[0]["RUSTWX_BASEMAP_DIR"] == str(companion)
    assert "render_warning" not in state
    assert not (plots._root(c.tmp_path, c.record["id"]) / plots.MAP_GAP).exists()


def test_a_gallery_drawn_with_no_map_files_says_so_in_the_job_status(prepared, tmp_path, monkeypatch):
    """The gallery still draws, and its status says what the pictures lack
    and the command that restores the files, until the gallery is done."""
    from test_render_basemap_delivery import wheel_with_companion
    wheel_with_companion(tmp_path, monkeypatch, maps=False)
    c = prepared.case
    state = plots.work_once(c.tmp_path, c.record["id"])
    assert len(prepared.renders) == 1, "a missing map is a note, never a refusal to draw"
    warning = state["render_warning"]
    assert "no coastlines, borders or state lines" in warning
    assert "pip install --force-reinstall recast-woof-data" in warning
    c.status["state"] = "completed"
    done = plots.work_once(c.tmp_path, c.record["id"])
    assert done["done"] and done["render_warning"] == warning
    # The job's own gallery status is the one `woof remote status` reads.
    status = json.loads((plots._root(c.tmp_path, c.record["id"]) / "status.json").read_text(encoding="utf-8"))
    assert status["render_warning"] == warning


def test_a_sections_only_run_draws_no_gallery_of_default_maps(prepared):
    c = prepared.case
    c.record["products"] = "xsec:wa"
    summary = plots.work_once(c.tmp_path, c.record["id"])
    assert summary["state"] == "no_map_products" and summary["note"] == viewer.NO_MAP_PRODUCTS_NOTE
    assert summary["selection_products"] == [] and not prepared.renders
    request = {**c.request, "action": "native-plots", "sequence": 1, "domain": 1}
    value = plots.catalog(request, c.tmp_path)
    assert value["selection_basis"] == viewer.NO_MAP_PRODUCTS_NOTE
    # The flag a terminal reads to answer with that note, instead of saying
    # the gallery is still being prepared for a run that will never draw one.
    assert value["map_products"] is False and value["waiting"] is True



def test_explicit_theme_and_auto_layout_reach_the_compact_renderer(prepared):
    c = prepared.case
    generic = plots.render_selection(c.record)
    assert generic["width"] == 1200 and generic["height"] == 900
    assert "theme" not in generic and "layout" not in generic
    assert generic["render_id"] == ra._sha(ra._encoded({
        "selection_id": generic["selection_id"], "width": 1200, "height": 900}))
    public = plots.render_selection(c.record, {"theme": "woof-light", "layout": "auto"})
    assert public["width"] is None and public["height"] is None
    assert public["render_id"] != generic["render_id"]
    plots.work_once(c.tmp_path, c.record["id"], selection=public)
    request = prepared.renders[-1]
    assert request["theme"] == "woof-light"
    assert request["width"] is None and request["height"] is None
    assert len(prepared.native.calls) == 1, "the theme reuses the prepared compact store"


def test_remote_theme_and_layout_are_explicit_parser_options():
    from woof.cli import build_parser
    args = build_parser().parse_args([
        "remote", "sync-native-plots", "--host", "test-host-1", "--python", "/python",
        "--workspace", "/work", "--job", "job-1", "--sequence", "3", "--cache-root", "cache",
        "--theme", "/site/theme.json", "--layout", "auto"])
    assert plots.request_options(args) == {"theme": "/site/theme.json", "layout": "auto"}


@pytest.mark.parametrize("presentation, message", [
    ({"layout": "auto", "width": 800}, "omit width and height"),
    ({"layout": "other"}, "auto or fixed"),
    ({"theme": " "}, "nonblank"),
    ({"theme": {"extends": "woof-light"}}, "nonblank"),
])
def test_remote_presentation_options_refuse_ignored_or_invalid_values(presentation, message):
    with pytest.raises(ValueError, match=message):
        plots.render_selection({"products": "2m_temperature"}, presentation)


def test_theme_edits_and_parent_assets_get_new_gallery_identities(tmp_path):
    parent = tmp_path / "parent.json"
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = child_dir / "site.json"
    font = tmp_path / "font.ttf"
    font.write_bytes(b"font content A; configuration fingerprint fixture")
    parent.write_text(json.dumps({"extends": "woof-light", "fonts": {"regular": "font.ttf"}}))
    child.write_text(json.dumps({"extends": "../parent.json"}))
    record = {"products": "2m_temperature"}
    request = {"theme": str(child)}
    first = plots.render_selection(record, request)
    assert plots.render_selection(record, request)["render_id"] == first["render_id"]
    parent.write_text(json.dumps({"extends": "woof-dark", "fonts": {"regular": "font.ttf"}}))
    second = plots.render_selection(record, request)
    assert second["render_id"] != first["render_id"]
    with pytest.raises(ValueError, match="changed after gallery selection"):
        plots._check_theme_selection(first)
    font.write_bytes(b"font content B; configuration fingerprint fixture")
    third = plots.render_selection(record, request)
    assert third["render_id"] != second["render_id"]
    child.write_text(json.dumps({"extends": "../parent.json", "text": {"model_label": "PUBLIC"}}))
    fourth = plots.render_selection(record, request)
    assert fourth["render_id"] != third["render_id"]


def test_a_remote_theme_parent_cycle_is_bounded(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps({"extends": "second.json"}))
    second.write_text(json.dumps({"extends": "first.json"}))
    with pytest.raises(ValueError, match="deeper than eight themes"):
        plots.render_selection({"products": "2m_temperature"}, {"theme": str(first)})
