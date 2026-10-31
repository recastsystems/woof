"""The RRFS route's Python stream XML parser, timelevel clock and file IDs (no native library).

CPU tests. The native comparisons run when their artifacts are supplied:
RRFS_NATIVE_STREAM_LIBRARY (the fork's capture library: ezxml + extract_stream_interval) and
RRFS_STREAM_FILES (an os.pathsep list of streams files or directories to replay).
"""
import ctypes
import os
import sys
from pathlib import Path

import pytest

from woof.hex.mpasphys.rrfs.stream_xml_rrfs import (extract_stream_interval, first_value_xml, native_file_id,
                                                   parse_xml, stream_tree, timelevel_query)


def test_first_attribute_wins_like_ezxml_attr():
    text = ('<streams><!-- a="1" a="2" --><stream name="x" filename_interval="1:00:00" type="output" '
            'filename_interval="none" output_interval="6:00:00"/></streams>')
    filtered = first_value_xml(text)
    assert 'filename_interval="none"' not in filtered
    assert '<!-- a="1" a="2" -->' in filtered
    import xml.etree.ElementTree as ET
    stream = ET.fromstring(filtered).find("stream")
    assert list(stream.attrib.items()) == [("name", "x"), ("filename_interval", "1:00:00"), ("type", "output"),
                                           ("output_interval", "6:00:00")]


@pytest.mark.parametrize("text", ['<a b="1" b=\'2\'>x</a>', '<a\nb="1"\tb="2"/>', '<a b="1" c="x>y" b="3" >t</a>',
                                  "<a/>", "<a></a>", '<?xml version="1.0"?><a b="&amp;"/>',
                                  "<a><![CDATA[<b c='1' c='2'>]]></a>"])
def test_filter_keeps_strict_xml(text):
    import xml.etree.ElementTree as ET
    ET.fromstring(first_value_xml(text))


@pytest.mark.parametrize("text", ['<a b="1"', '<a b=1/>', '<a b/>', "<!-- open", '<a b="1>'])
def test_filter_refuses_what_it_cannot_delimit(text):
    with pytest.raises(ValueError):
        first_value_xml(text)


def _streams(tmp_path, body):
    path = tmp_path / "streams.atmosphere"
    path.write_text("<streams>" + body + "</streams>")
    return path


def test_interval_references_and_filename_defaults(tmp_path):
    path = _streams(tmp_path, '''
      <immutable_stream name="input" type="input" filename_template="init.nc" input_interval="initial_only"/>
      <stream name="history" type="output" filename_template="h.nc" output_interval="6:00:00"/>
      <stream name="diag" type="output" filename_template="d.nc" output_interval="stream:history:output_interval"/>
      <stream name="lbc" type="input" filename_template="l.nc" input_interval="3:00:00" filename_interval="input_interval"/>
      <stream name="tl" type="output" filename_template="t.nc" output_timelevels="0-3-15m"/>
      <stream name="once" type="output" filename_template="o.nc" output_interval="initial_only"
              filename_interval="output_interval"/>''')
    tree = stream_tree(path, library=None)
    got = {el.get("name"): el.get("filename_interval") for el in tree}
    assert got == {"input": "none", "history": "6:00:00", "diag": "6:00:00", "lbc": "3:00:00",
                   "tl": "output", "once": "none"}
    assert tree.find("stream[@name='diag']").get("output_interval") == "stream:history:output_interval"


@pytest.mark.parametrize("value,message", [
    ("stream:diag:output_interval", "references itself"),
    ("stream:history:record_interval", "invalid attribute"),
    ("stream:nowhere:output_interval", "undefined stream"),
    ("stream:history:input_interval", "undefined attribute"),
    ("stream:alias:output_interval", "unexpandable"),
    ("stream:nocolon", "invalid attribute")])
def test_extract_stream_interval_refusals(tmp_path, value, message):
    root = parse_xml(_streams(tmp_path, f'''
      <stream name="history" type="output" filename_template="h.nc" output_interval="6:00:00"/>
      <stream name="alias" type="output" filename_template="a.nc" output_interval="stream:history:output_interval"/>
      <stream name="diag" type="output" filename_template="d.nc" output_interval="{value}"/>'''))
    with pytest.raises(ValueError, match=message):
        extract_stream_interval(value, "output_interval", "diag", root)


def test_timelevels_known_words():
    q = timelevel_query("0-3-15m 4-24", 3600)
    assert q["matched"] and q["interval_seconds"] == 900.0
    assert q["words"][0] == [0, 0x40400000, 0x41700000]  # 0 h, 3 h, 15 min as binary32
    q = timelevel_query("1 2 3", 5400)
    assert not q["matched"] and q["future_seconds"] == 7200.0 and q["next_seconds"] is None


@pytest.mark.parametrize("spec", ["", " ", "0--2", "-2", "2-", "0-2-", "1H", "xm", "1h2", "0-1-1-1",
                                  " ".join(["0"] * 33)])
def test_timelevel_refusals(spec):
    with pytest.raises(ValueError):
        timelevel_query(spec, 0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the GNU libc rand() referee is Linux-only")
@pytest.mark.parametrize("seed", [0, 1, 7, 1234567, 2**31 - 1, 2**31, 2**32 - 1])
@pytest.mark.parametrize("ordinal", [0, 1, 7, 31])
def test_file_ids_follow_gnu_rand(seed, ordinal):
    libc = ctypes.CDLL("libc.so.6")
    libc.srand(ctypes.c_uint(seed))
    for _ in range(10 * ordinal):
        libc.rand()
    expected = "".join("0123456789abcdefghijklmnopqrstuvwxyz"[libc.rand() % 36] for _ in range(10))
    assert native_file_id(seed, ordinal) == expected


def _shipped():
    files = []
    for entry in filter(None, os.environ.get("RRFS_STREAM_FILES", "").split(os.pathsep)):
        root = Path(entry)
        files += [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.is_file() and p.name.startswith("streams.atmosphere"))
    return files


def test_every_shipped_streams_file_equals_native_ezxml():
    library = os.environ.get("RRFS_NATIVE_STREAM_LIBRARY")
    files = _shipped()
    if not library or not files:
        pytest.skip("native RRFS capture library or shipped streams files not supplied")
    from rrfs_native_stream_tree import stream_tree as native_stream_tree

    def flat(element, depth=0):
        yield depth, element.tag, list(element.attrib.items())
        for child in element:
            yield from flat(child, depth + 1)

    for path in files:
        assert list(flat(stream_tree(path))) == list(flat(native_stream_tree(path, library))), path
