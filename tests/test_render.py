"""``woof render`` matplotlib-engine product tests -- CPU-only.

These pin the matplotlib engine explicitly: they are the fallback
engine's product contract, and must hold identically whether or not
the rust renderer is built (tests/test_render_rust.py covers that
engine and the auto selection).

The fixture is written by the project's own ``WrfoutWriter`` (WRF
dimensions, attributes, and Times records), so the file under test is
what ``woof run`` produces, and the wrf package (pip ``wrf-rust``) reads
it exactly as it reads a production frame.  Assertions check that the
requested PNGs exist, are non-empty, and decode to plausible image
dimensions -- not that any pixel is pretty.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "wrf", reason="woof render requires the wrf package (wrf-rust)")

import woof.cli as cli
from woof.io.wrfout import WrfoutWriter
from woof.render import (DEFAULT_SOURCE_LABEL, domain_token, parse_products,
                          parse_timeidx, plot_context, resolution_token,
                          spacing_label)

from test_render_layout import _NX, _NY, _NZ, _STAMPS, _frame


@pytest.fixture(scope="module")
def wrfout(tmp_path_factory):
    path = tmp_path_factory.mktemp("render") / \
        "wrfout_d02_1974-04-03_18-00-00.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        for index, stamp in enumerate(_STAMPS):
            writer.write_frame(stamp, _frame(seed=7 + index))
    return path


def _png_shape(path):
    import matplotlib.image
    return matplotlib.image.imread(path).shape


def _spy_wrf(monkeypatch, calls):
    import woof.render as render
    import wrf as real_wrf

    class SpyWrf:
        def __getattr__(self, name):
            return getattr(real_wrf, name)

        @staticmethod
        def getvar(wf, name, timeidx=None, **kwargs):
            calls.append((name, kwargs.get("units")))
            return real_wrf.getvar(wf, name, timeidx=timeidx, **kwargs)

    monkeypatch.setattr(render, "_import_wrf", lambda: SpyWrf())


def test_render_all_products_all_frames(wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.rglob("*.png"))
    expected = sorted(
        f"{product}_d02-1km_{stamp.replace(':', '-')}.png"
        for product in ("refl", "t2", "wind10", "precip", "olr")
        for stamp in _STAMPS)
    assert produced == expected
    for png in out.rglob("*.png"):
        assert png.stat().st_size > 5_000, png.name
        height, width, _ = _png_shape(png)
        assert height >= 400 and width >= 600, (png.name, height, width)


def test_render_product_and_frame_selection(wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--products", "refl,t2",
                   "--timeidx", "1", "--out", str(out), "--dpi", "72"])
    assert rc == 0
    produced = sorted(p.name for p in out.rglob("*.png"))
    stamp = _STAMPS[1].replace(":", "-")
    assert produced == sorted(
        [f"refl_d02-1km_{stamp}.png", f"t2_d02-1km_{stamp}.png"])


def _wrfout_without(tmp_path, *drop: str, name="wrfout_d01_x.nc"):
    """A one-frame wrfout with ``drop`` removed from the field set."""

    path = tmp_path / name
    fields = _frame(seed=3)
    for field in drop:
        del fields[field]
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], fields)
    return path


def test_a_field_the_file_does_not_carry_skips_that_product_at_rc_0(
        tmp_path, capsys):
    """The cold-start frame's shape: no REFL_10CM, and that is not an error.

    A history frame written before any microphysics call carries no
    ``REFL_10CM`` -- a registered deviation, not a fault -- and this used
    to come back as ``render FAIL`` with exit 1, which stopped the whole
    ``woof go`` chain on a forecast that had completed and a render that
    had drawn every product the file could support.

    The contract now: the products whose declared inputs are present are
    drawn, the one whose input is absent is SKIPPED BY NAME, and the exit
    code is 0.  Named, because a skip nobody prints is the silent success
    this project refuses.
    """
    path = _wrfout_without(tmp_path, "REFL_10CM")
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "refl,t2", "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.rglob("*.png")]
    assert produced == [f"t2_d01-1km_{_STAMPS[0].replace(':', '-')}.png"]

    err = capsys.readouterr().err
    assert "render FAIL" not in err
    assert "skipped" in err
    # The default layer names the PRODUCT -- a bare count is not
    # actionable -- and says which of the two things this is.
    assert "refl" in err
    assert "not a failure" in err
    # And the FIELD the product lacked, which is what a reader needs to
    # act on; the file it was recorded against is not on the line.  This
    # pin used to be the reverse ("REFL_10CM" not in err): the note now
    # gives each product its own first reason, and keeps the file and
    # the per-item rows behind --explain.
    note = [line for line in err.splitlines()
            if line.startswith("note: render skipped")]
    assert len(note) == 1, err
    assert "refl: the file carries no REFL_10CM" in note[0], note[0]
    # (The --explain hint below the note repeats the command line, file
    # included; that is the way in, not the reason.)
    assert path.name not in note[0], "the file belongs behind --explain"
    assert "skipped refl:" not in err, "per-item rows belong behind --explain"

    # --explain restores the per-item evidence: which file, which frame,
    # which field.
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "refl,t2", "--out", str(tmp_path / "png2"),
                   "--explain"])
    assert rc == 0
    explained = capsys.readouterr().err
    assert "REFL_10CM" in explained
    assert f"skipped refl: {path}[0] carries no REFL_10CM" in explained


def test_a_skip_still_exits_1_when_it_leaves_nothing_drawn(tmp_path):
    """The arm that keeps a skip from becoming a silent success.

    Ask for exactly the product whose input is absent and no image is
    produced at all.  A render that drew nothing is a failed render
    however accurate its reason, so the exit code stays 1.
    """
    path = _wrfout_without(tmp_path, "REFL_10CM")
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "refl", "--out", str(out)])
    assert rc == 1
    assert not list(out.rglob("*.png"))


def test_the_nothing_drawn_notice_does_not_deny_the_nonzero_exit(
        tmp_path, capsys):
    """The skip notice must not contradict the exit code it ships with.

    Ask for exactly one product whose input is absent: the render exits 1
    because nothing was drawn (the arm above).  The notice used to append
    "That is not a failure and does not change the exit code" -- true when
    other products rendered, FALSE here, where the skips are precisely why
    the exit code is 1.  A user chasing that sentence concluded the tool
    was broken (a real complaint: lifted_index requested from a wrfout,
    rendered=0, exit 1, and a note insisting nothing failed).

    The notice still names the product either way; what changes is the
    verdict half: with nothing drawn it must SAY nothing was drawn and
    that the exit code reflects it.
    """
    path = _wrfout_without(tmp_path, "REFL_10CM")
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "refl", "--out", str(tmp_path / "png")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "refl" in err
    assert "not a failure" not in err
    assert "does not change the exit code" not in err
    assert "nothing rendered" in err

    # The mixed outcome keeps the reassuring sentence: skips beside a
    # drawn product still exit 0 and still say so.
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "refl,t2", "--out", str(tmp_path / "png2")])
    assert rc == 0
    err = capsys.readouterr().err
    assert "not a failure" in err


def test_a_product_whose_inputs_are_present_still_fails_loudly(
        tmp_path, monkeypatch, capsys):
    """Negative control: the skip branch must not swallow a real fault.

    Every declared input is in the file, so the availability table has
    no opinion; the product raises anyway.  That is an unregistered
    failure and it must stay one -- ``render FAIL``, exit 1 -- or this
    change would be the fourth warn-and-drop to disable a deliberate
    refusal on this release line.
    """
    from woof import render as render_module

    path = _wrfout_without(tmp_path)          # nothing dropped
    out = tmp_path / "png"

    def explode(*_args, **_kwargs):
        raise RuntimeError("colormap exploded")

    monkeypatch.setitem(render_module.PRODUCTS, "t2", explode)
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "refl,t2", "--out", str(out)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "render FAIL" in err
    assert "colormap exploded" in err


def test_the_availability_table_is_the_one_the_listing_prints(tmp_path):
    """One table, two readers -- the listing and the render decision.

    The skip exists because ``--list-products`` already knew the answer
    and the render loop was asking an exception instead.  If these two
    ever disagree, a file would list a product as ``missing-fields`` and
    then fail the run for not having it.
    """
    from woof import render as render_module

    path = _wrfout_without(tmp_path, "REFL_10CM")
    rows = {slug: status
            for slug, _kind, status, _detail
            in render_module._list_products_matplotlib(path)}
    assert rows["refl"] == "missing-fields"
    assert rows["t2"] == "renderable"

    present = render_module._file_variables(path)
    assert render_module.missing_declared_inputs(present, "refl") \
        == ["REFL_10CM"]
    assert render_module.missing_declared_inputs(present, "t2") == []


def test_render_timeidx_out_of_range_is_an_error(wrfout, tmp_path):
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--products", "t2",
                   "--timeidx", "9", "--out", str(tmp_path / "png")])
    assert rc == 1
    assert not list((tmp_path / "png").rglob("*.png"))


def test_resolution_tokens_are_trimmed_km_or_integer_metres():
    """Format pin, shared with the rust engine's own unit tests."""

    assert resolution_token(3_000.0) == "3km"
    assert resolution_token(12_000.0) == "12km"
    assert resolution_token(1_000.0) == "1km"
    # A 3:1 nest of a 12 km parent, twice over.
    assert resolution_token(1_333.3333) == "1.333km"
    assert resolution_token(1_500.0) == "1.5km"
    # Sub-kilometre reads as metres, not as 0.111 km.
    assert resolution_token(111.1111) == "111m"
    assert resolution_token(500.0) == "500m"
    # The boundary rounds before it chooses a unit.
    assert resolution_token(999.6) == "1km"
    assert resolution_token(999.4) == "999m"
    # Nothing usable is never guessed at.
    assert resolution_token(None) is None


def test_spacing_labels_spell_the_same_number():
    assert spacing_label(3_000.0) == "Δx 3 km"
    assert spacing_label(1_333.3333) == "Δx 1.333 km"
    assert spacing_label(111.1111) == "Δx 111 m"
    assert spacing_label(None) is None


def test_domain_token_pairs_domain_with_resolution():
    assert domain_token("d02", 3_000.0) == "d02-3km"
    assert domain_token("d05", 111.1111) == "d05-111m"
    # A file declaring no DX still separates its nests by domain.
    assert domain_token("d02", None) == "d02"


def test_plot_context_is_the_second_title_line():
    """Grid identity, spacing, valid time, and who produced it -- the
    matplotlib mirror of the rust engine's subtitle."""

    assert plot_context("d02", 3_000.0, "1974-04-03_18:00:00",
                        DEFAULT_SOURCE_LABEL) == (
        "d02-3km | Δx 3 km | valid 1974-04-03_18:00:00 | WOOF-ArWen")
    # No DX in the file: the segment is absent, never invented.
    assert plot_context("d02", None, "1974-04-03_18:00:00", "WRF-ARW") == (
        "d02 | valid 1974-04-03_18:00:00 | WRF-ARW")


def test_titles_carry_the_spacing_and_the_source_label(wrfout, tmp_path,
                                                       monkeypatch):
    """The title text reaches matplotlib, not just the filename."""

    import woof.render as render

    titles: list[str] = []
    real_finish = render._finish

    def spy(fig, axis, mappable, *, title, **kwargs):
        titles.append(title)
        return real_finish(fig, axis, mappable, title=title, **kwargs)

    monkeypatch.setattr(render, "_finish", spy)
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "t2", "--timeidx", "0",
                   "--out", str(tmp_path / "png")])
    assert rc == 0
    # The default label carries the EXECUTING version, so it is asked
    # for rather than transcribed -- a hardcoded "ArWen 1.8.7" here
    # would have to be edited at every cut, and the property under test
    # is that the label reaches the title, not what this release is
    # numbered.
    assert titles == [
        "2 m temperature\nd02-1km | Δx 1 km | "
        f"valid 1974-04-03_18:00:00 | {render.default_source_label()}"]
    # ... and that it really did carry a version, not the bare brand.
    assert render.default_source_label().startswith("WOOF-ArWen ")

    titles.clear()
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "t2", "--timeidx", "0",
                   "--source-label", "WRF-ARW 4.6.1",
                   "--out", str(tmp_path / "png2")])
    assert rc == 0
    assert titles[0].endswith("| WRF-ARW 4.6.1"), titles


def test_parse_products():
    assert parse_products("all") == ("refl", "t2", "wind10", "precip", "olr")
    assert parse_products("precip,refl") == ("precip", "refl")
    assert parse_products("refl,refl") == ("refl",)
    with pytest.raises(ValueError, match="unknown product"):
        parse_products("refl,uh25")
    with pytest.raises(ValueError, match="no products"):
        parse_products(",")
    # A section term is named whole, level list and closing term included.
    with pytest.raises(ValueError, match=r"'xsec:QCLOUD=0\.01,0\.1/wa'"):
        parse_products("refl,xsec:QCLOUD=0.01,0.1/wa")


def test_parse_timeidx():
    assert parse_timeidx("all") is None
    assert parse_timeidx("3") == 3
    with pytest.raises(ValueError, match="integer or 'all'"):
        parse_timeidx("last")
    with pytest.raises(ValueError, match="non-negative"):
        parse_timeidx("-1")


def test_unknown_product_is_exit_code_2(wrfout, tmp_path):
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--products", "vorticity",
                   "--out", str(tmp_path / "png")])
    assert rc == 2


def test_olr_draws_from_the_field_the_frame_carries(tmp_path, monkeypatch):
    """The OLR panel is a retrieval, not a computation.

    wrfout frames have carried ``OLR`` since 1.5.1 and the fallback had no
    renderer for it.  This holds the two things that make the new one a
    product rather than a picture: the value comes out of the file through
    ``wrf.getvar`` with no unit argument -- W m-2 is what the field is, and
    a brightness temperature would be a derived quantity this site is not
    allowed to derive -- and the panel is labeled in those units.
    """
    import woof.render as render

    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=13))

    calls: list[tuple[str, object]] = []

    _spy_wrf(monkeypatch, calls)
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "olr", "--out", str(out)])
    assert rc == 0
    drawn = sorted(out.rglob("*.png"))
    assert [p.name for p in drawn] == [
        f"olr_d01-1km_{_STAMPS[0].replace(':', '-')}.png"]
    assert drawn[0].stat().st_size > 5_000
    # Retrieved by name, in the field's own units, with no conversion.
    assert ("OLR", None) in calls, calls


def test_olr_is_drawn_as_inverted_grayscale_infrared(tmp_path, monkeypatch):
    """Cold tops bright, warm surface dark -- the IR imagery convention.

    Asserted at the colormap the panel is handed, because that is the one
    decision that makes this a synthetic-satellite product rather than a
    generic fill: ``gray_r`` maps LOW emitted flux (deep cold cloud) to
    white and HIGH flux (clear warm surface) to black.  A plain ``gray``
    would draw a photographic negative of every satellite loop a forecaster
    has ever read.
    """
    import woof.render as render

    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=17))

    seen: list[dict] = []
    plt = render._pyplot()
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, axis = real_subplots(*args, **kwargs)
        real_pcolormesh = axis.pcolormesh

        def capture(*a, **kw):
            seen.append(kw)
            return real_pcolormesh(*a, **kw)

        axis.pcolormesh = capture
        return fig, axis

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "olr", "--out", str(tmp_path / "png")])
    assert rc == 0
    assert seen and seen[0].get("cmap") == "gray_r", seen


def test_a_frame_without_olr_skips_the_product_rather_than_failing(
        tmp_path, capsys):
    """Radiation off means no OLR, and that is a report, not a fault.

    ``OLR`` is present exactly when the attached longwave scheme produced
    it, so the absence is a legitimate configuration and has to travel the
    same skip path REFL_10CM's absence does.
    """
    from woof import render as render_module

    path = _wrfout_without(tmp_path, "OLR")
    rows = {slug: status
            for slug, _kind, status, _detail
            in render_module._list_products_matplotlib(path)}
    assert rows["olr"] == "missing-fields"

    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "olr,t2", "--out", str(out)])
    assert rc == 0
    assert [p.name for p in out.rglob("*.png")] == [
        f"t2_d01-1km_{_STAMPS[0].replace(':', '-')}.png"]
    err = capsys.readouterr().err
    assert "olr" in err and "not a failure" in err


def test_wind10_fallback_is_native_units_via_getvar(tmp_path, monkeypatch):
    """No SINALPHA/COSALPHA: the wind panel falls back to raw U10/V10 in
    the model's native m/s -- every retrieval through wrf.getvar with no
    unit arguments and no local conversion factor."""
    import woof.render as render

    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    fields = _frame(seed=11)
    del fields["SINALPHA"]
    del fields["COSALPHA"]
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], fields)

    calls: list[tuple[str, object]] = []

    _spy_wrf(monkeypatch, calls)
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path), "--products", "wind10",
                   "--out", str(out)])
    assert rc == 0
    assert list(out.rglob("wind10_*.png"))
    # The earth-rotated primary was attempted (in knots, via getvar) ...
    assert ("uvmet10", "kt") in calls
    # ... and the fallback retrieved raw fields with NO unit arguments:
    # native m/s, labeled as such, no hand-rolled conversion.
    for name in ("U10", "V10", "wspd10"):
        assert (name, None) in calls, calls
        assert (name, "kt") not in calls, calls
    # The kt-per-(m/s) constant must not exist anywhere in the module.
    from pathlib import Path as _Path
    assert "1.9438" not in _Path(render.__file__).read_text()


def test_the_matplotlib_engine_announces_itself_as_a_workaround(
        monkeypatch, capsys):
    """An engine that does not say so is indistinguishable from the real
    thing until someone counts the products.

    A pilot spent a whole session on the four-product matplotlib path
    without noticing the 151-product rust catalog was simply not built.

    It is a WORKAROUND rather than a fallback since the render law's
    one-fallback clause was enforced (audit F7): nothing degrades into
    it, it is reachable only by typing ``--engine matplotlib``, and
    "Fixed means default" (project ruling, 2026-08-10) requires an opt-in remedy to
    be REPORTED as a workaround on every run -- so the explicit request
    is exactly the case that must not be silent.  It used to be.
    """
    import argparse
    from pathlib import Path
    from woof import render as render_module

    assert render_module.matplotlib_workaround_notice("rust") is None

    notice = render_module.matplotlib_workaround_notice("matplotlib")
    assert notice is not None
    # ONE line: a notice that scrolls is a notice that gets skimmed.
    assert "\n" not in notice, notice
    # The project's own word for a reachable-but-not-default path.
    assert notice.startswith("WORKAROUND:")
    # The engine actually used ...
    assert "engine matplotlib" in notice
    assert "render law" in notice
    # ... what that costs, against the rust catalog ...
    assert f"{len(render_module.PRODUCTS)} of the rust catalog's " \
           f"{render_module._RUST_CATALOG_PRODUCTS} products" in notice
    # ... and, in a checkout, the exact command that fixes it.
    from woof import bridges, rustwx
    assert bridges.sources_present(rustwx.RUSTWX_CRATE_RELATIVE)
    assert rustwx.CARGO_BUILD_HINT in notice
    assert "cargo build --release --locked --offline" in notice

    # It reaches stderr on a real invocation, before anything else.
    args = argparse.Namespace(
        pair=None, wrfout=[Path("nonexistent.nc")], timeidx="0",
        size="1200x900", engine="matplotlib", products="all",
        list_products=False, out=Path("out"), dpi=110, heavy=False,
        layout=render_module.render_layout.DEFAULT_LAYOUT,
        source_label=render_module.DEFAULT_SOURCE_LABEL)
    monkeypatch.setattr(render_module, "render_wrfouts",
                        lambda *a, **k: ([], [], []))
    render_module.render_main(args)
    captured = capsys.readouterr()
    # (the stub writes nothing, so the exit code is the empty-output one;
    # what is under test is that the notice was printed, and printed
    # before any product line.)
    assert "WORKAROUND:" in captured.err
    assert "render law" in captured.err


def test_the_workaround_notice_is_install_aware_in_one_line(monkeypatch):
    """A wheel install has no `tools/rustwx` to cd into.

    The 1.0.1 remedy contract stopped doctor sending pip users to a
    directory a wheel does not carry; this second call site assembled its
    own string and kept doing it.  It is bound to ONE physical line, and
    the pip-shape answer is a whole bootstrap -- so the accurate one-line
    composition names `woof doctor` rather than inlining a command that
    cannot work here.
    """
    from woof import bridges, render as render_module, rustwx

    # Checkout shape: the one-liner, unchanged.
    monkeypatch.setattr(bridges, "sources_present", lambda *a, **k: True)
    checkout = render_module.matplotlib_workaround_notice("matplotlib")
    assert "\n" not in checkout
    assert rustwx.CARGO_BUILD_HINT in checkout

    # Wheel shape: still one line, and it no longer names the directory
    # that is not there.
    monkeypatch.setattr(bridges, "sources_present", lambda *a, **k: False)
    wheel = render_module.matplotlib_workaround_notice("matplotlib")
    assert wheel is not None
    assert "\n" not in wheel, wheel
    assert "cd tools" not in wheel, wheel
    assert "woof doctor" in wheel
    # Still says what the workaround costs, in both shapes.
    for text in (checkout, wheel):
        assert "engine matplotlib" in text
        assert f"{len(render_module.PRODUCTS)} of the rust catalog's" in text


def test_sources_present_answers_about_the_crate_it_is_asked_about(
        monkeypatch, tmp_path):
    """It used to answer for tools/grib1_bridge whatever it was asked.

    A tree carrying one crate and not the other therefore got a `cd` into
    the missing one -- the exact class of bug the install-aware remedies
    exist to prevent.
    """
    from woof import bridges

    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    (tmp_path / "tools" / "grib1_bridge").mkdir(parents=True)
    (tmp_path / "tools" / "grib1_bridge" / "Cargo.toml").write_text("[package]\n")

    assert bridges.sources_present("tools/grib1_bridge") is True
    assert bridges.sources_present("tools/rustwx") is False
    # The default is unchanged for every existing caller.
    assert bridges.sources_present() is True


def test_sources_present_is_false_for_a_crate_folder_holding_only_package_data(
        monkeypatch, tmp_path):
    """A pip install carries tools/hydro_world/data (package data), not the crate.

    Breakage (2.9.4 R11, the packet's Linux wheel): the doctor took that data
    folder for a checkout, so rw_hydro_world and rw_hydro_params lost their
    on-demand row and got a `cargo build` remedy inside site-packages.
    """
    from woof import bridges

    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path)
    data = tmp_path / "tools" / "hydro_world" / "data"
    data.mkdir(parents=True)
    (data / "sources.v1.json").write_text("{}")

    assert bridges.sources_present("tools/hydro_world") is False
    (tmp_path / "tools" / "hydro_world" / "Cargo.toml").write_text("[package]\n")
    assert bridges.sources_present("tools/hydro_world") is True


# ---------------------------------------------------------------------------
# Degradation: identical to the rust engine's, or it is not one behaviour
# ---------------------------------------------------------------------------
# The mirror of tests/test_render_rust.py's
# test_unknown_domain_identity_keeps_the_native_grid_token.  Matplotlib
# guessed `d01` for a file carrying neither GRID_ID nor a wrfout_dNN
# name, which labelled an anonymous input as the parent domain on no
# evidence and -- because the token is what separates one nest's output
# names from another's -- let two such inputs at one valid time
# overwrite each other, the exact failure this release exists to end.

def test_domain_token_degrades_exactly_as_the_rust_engine_does():
    """The three steps, stated once, asserted on the matplotlib side."""

    from woof.render import NATIVE_GRID_SLUG

    # 1. Identity + usable DX -> dNN-resolution.
    assert domain_token("d02", 3_000.0) == "d02-3km"
    # 2. Identity, no usable DX -> bare dNN; the domain alone still
    #    separates the nests, which is the token's whole job.
    assert domain_token("d02", None) == "d02"
    # 3. No identity -> the generic slug, and NO resolution appended: a
    #    resolution is not an identity.
    assert domain_token(None, 3_000.0) == NATIVE_GRID_SLUG
    assert domain_token(None, None) == NATIVE_GRID_SLUG
    # Spelled the same as rust spells it, so one word means one thing.
    assert NATIVE_GRID_SLUG == "native_grid"


def test_unknown_domain_identity_keeps_the_native_grid_token(tmp_path):
    """No GRID_ID, no wrfout_dNN name: the token degrades, never guesses."""

    path = tmp_path / "some_model_output.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=21))

    import netCDF4
    with netCDF4.Dataset(path) as ds:
        assert "GRID_ID" not in ds.ncattrs(), (
            "the fixture must be anonymous for this test to mean anything")

    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "t2", "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.rglob("*.png")]
    assert produced, "the anonymous-domain file rendered nothing"
    for name in produced:
        assert "native_grid" in name, name
        # The old guess, in either spelling.
        assert "_d01_" not in name and "_d01-" not in name, name


def test_two_anonymous_inputs_refuse_to_overwrite_each_other(tmp_path):
    """The collision the guessed `d01` used to hide.

    Both files are unidentifiable, so both resolve to the same
    `native_grid` token -- and at one valid time, for one product, to
    one output name.  Silently writing the second over the first is the
    v1.0.0 data-loss bug wearing a different token, so the second is
    refused with the reason and the remedy.
    """

    paths = []
    for index in (0, 1):
        path = tmp_path / f"anonymous_{index}.nc"
        with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                          dx=1000.0, dy=1000.0) as writer:
            writer.write_frame(_STAMPS[0], _frame(seed=31 + index))
        paths.append(str(path))

    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", *paths,
                   "--products", "t2", "--out", str(out)])
    # One PNG, and a loud non-zero exit -- never two inputs and one
    # silent survivor.
    assert rc == 1
    assert len(list(out.rglob("*.png"))) == 1


def test_a_grid_id_out_of_domain_range_is_not_identity(tmp_path):
    """`GRID_ID = 0` is not domain zero; it is no evidence.

    The rust side accepts 1..=99 and this must agree, or one engine
    invents a `d00` the other refuses.
    """

    from woof.render import _domain_tag

    path = tmp_path / "odd_grid_id.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0,
                      dy=1000.0, global_attrs={"GRID_ID": 0}) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=41))
    assert _domain_tag(path) is None

    named = tmp_path / "wrfout_d03_1974-04-03_18-00-00.nc"
    with WrfoutWriter(named, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0,
                      dy=1000.0, global_attrs={"GRID_ID": 0}) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=42))
    # GRID_ID unusable, so the filename is the evidence -- rust's
    # precedence exactly.
    assert _domain_tag(named) == "d03"
