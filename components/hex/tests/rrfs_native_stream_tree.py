"""The native RRFS stream XML tree, through the fork's own ezxml in its capture library (test oracle only).

This was ``woof.hex.mpasphys.rrfs.driver_rrfs.stream_tree``, the RRFS route's ``streams_parser`` until the
route took the native-free ``woof.hex.mpasphys.rrfs.stream_xml_rrfs.stream_tree`` (894 shipped streams files,
0 differing: tests/replay_rrfs_stream_tree.py). Product code no longer opens the native library to read a
streams file; the replays and native-comparison tests keep this as their referee.
"""


def stream_tree(path, library, *, pin_set=None):
    """Use pristine stream XML semantics; MPAS-Model-rrfs/src/external/ezxml/ezxml.h:49-63; MPAS-Model-rrfs/src/external/ezxml/ezxml.c:83-99,662-670,791-830,834-838.

    The published GSL streams repeat filename_interval. Native accepts this
    and ezxml_attr returns the first value. Retain the original input bytes;
    this copies stream declarations only, without processing weather fields.
    """
    import ctypes as ct
    import xml.etree.ElementTree as ET
    from woof.hex.mpasphys.native_backend import NativePhysics
    from woof.hex.mpasphys.rrfs.driver_rrfs import specification
    class Node(ct.Structure):
        pass
    pointer = ct.POINTER(Node)
    Node._fields_ = [('name', ct.c_char_p), ('attr', ct.POINTER(ct.c_char_p)),
                    ('text', ct.c_char_p), ('off', ct.c_size_t),
                    *[(name, pointer) for name in ('next', 'sibling', 'ordered', 'child', 'parent')],
                    ('flags', ct.c_short)]
    with NativePhysics(library, test_only=True, pin_set=pin_set or specification()['pin_set']) as native:
        parse, error, free = (getattr(native.lib, 'ezxml_' + name)
                             for name in ('parse_file', 'error', 'free'))
        parse.argtypes, parse.restype = [ct.c_char_p], pointer
        error.argtypes, error.restype = [pointer], ct.c_char_p
        free.argtypes, free.restype = [pointer], None
        root = parse(str(path).encode())
        if not root:
            raise OSError(f'native stream XML could not open {path}')
        try:
            problem = error(root)
            if problem:
                raise ValueError(problem.decode())
            def copy(node):
                """Copy declared metadata; MPAS-Model-rrfs/src/external/ezxml/ezxml.c:83-99; MPAS-Model-rrfs/src/external/ezxml/ezxml.h:49-63."""
                attrs, i = {}, 0
                while node.contents.attr[i]:
                    attrs.setdefault(node.contents.attr[i].decode(), node.contents.attr[i + 1].decode())
                    i += 2
                element = ET.Element(node.contents.name.decode(), attrs)
                child = node.contents.child
                while child:
                    element.append(copy(child))
                    child = child.contents.ordered
                return element
            tree = copy(root)
            # Complete native filename policy, including recurring defaults and
            # output_timelevels. Explicit duplicate attributes retain first-value
            # ezxml semantics above. Delegate interval aliases to pristine C.
            # MPAS-Model-rrfs/src/framework/xml_stream_parser.c:925-1023,1150-1209,1495-1560.
            extract = native.lib.extract_stream_interval
            extract.argtypes = [ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_char_p),
                                ct.c_char_p, pointer]
            extract.restype = ct.c_int
            for element in tree:
                if element.tag not in ("stream", "immutable_stream"):
                    continue
                intervals = {}
                for key in ("input_interval", "output_interval"):
                    raw = element.get(key)
                    if raw is None:
                        continue
                    resolved = ct.c_char_p(raw.encode())
                    status = extract(raw.encode(), key.encode(), ct.byref(resolved),
                                     element.get("name").encode(), root)
                    if status:
                        raise ValueError(f"native stream {element.get('name')} cannot resolve {key}")
                    intervals[key] = resolved.value.decode()
                def recurring(key):
                    """Native keyword predicate; MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1508-1529."""
                    raw = element.get(key)
                    return raw is not None and not any(token in raw for token in
                                                       ("initial_only", "final_only", "none"))
                interval = element.get("filename_interval")
                direction = element.get("type", "")
                if interval is None:
                    if "input" in direction and recurring("input_interval"):
                        interval = intervals["input_interval"]
                    elif "output" in direction and recurring("output_interval"):
                        interval = intervals["output_interval"]
                    elif (element.tag == "stream" and "output" in direction
                          and element.get("output_timelevels") is not None):
                        interval = "output"
                else:
                    for key in ("input_interval", "output_interval"):
                        if key in interval:
                            interval = intervals[key] if recurring(key) else None
                            break
                element.set("filename_interval", interval or "none")
            return tree
        finally:
            free(root)
