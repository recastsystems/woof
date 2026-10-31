"""Compare the port to Fortran emitted by the actual pinned MPAS C generator.

Set RRFS_NATIVE_REGISTRY_GENERATED to a directory containing full/ and dry/
outputs. Generation/build happens explicitly under a bounded build policy;
this test never builds or substitutes a Python Registry parser as its oracle.
"""
import os
from pathlib import Path
import re

import pytest

from woof.hex.streams_rrfs import _REGISTRY, field_definition, registry_stream, package_active, _all_packages


def _generated(physics):
    root=os.environ.get('RRFS_NATIVE_REGISTRY_GENERATED')
    if not root:pytest.skip('MPAS native registry generator artifacts not supplied')
    return Path(root)/('full' if physics else 'dry')


def _attrs(block):
    result={}
    for name,value in re.findall(r"call mpas_add_att\([^\n]*? % attList, '([^']+)', (.*?)\)$",block,re.M):
        if value.startswith("'"):value=value[1:-1].replace("''", "'")
        else:value=float(value) if '.' in value else int(value)
        assert name not in result or result[name]==value
        result[name]=value
    return result


def _native_fields(path):
    source=path.read_text().replace('&\n&','').replace('&\n','')
    result={}
    pieces=re.split(r'^! Define (variable|var array) (\w+)\n',source,flags=re.M)
    for kind,name,body in zip(pieces[1::3],pieces[2::3],pieces[3::3]):
        pointer=re.search(r'allocate\(\s*([ricl])\d',body).group(1)
        primitive={'r':'real','i':'integer','c':'text','l':'logical'}[pointer]
        dims=dict((int(i),d) for i,d in re.findall(r"% dimNames\((\d+)\) = '([^']+)'",body))
        dimensions=[d for i,d in sorted(dims.items())]
        if kind=='var array':dimensions=dimensions[1:]
        if primitive=='text':dimensions.insert(0,'StrLen')
        if re.search(r'% hasTimeDimension = \.true\.',body):dimensions.append('Time')
        if kind=='variable':
            result[name]={'type':primitive,'dimensions':dimensions,'attrs':_attrs(body)}
        else:
            constituents=re.split(r"call mpas_pool_get_dimension\(newSubPool, 'index_(\w+)', const_index\)",body)
            for variable,block in zip(constituents[1::2],constituents[2::2]):
                variable=re.search(r"% constituentNames\(const_index\) = '([^']+)'",block).group(1)
                # Attributes are attached once at the first time level; later
                # levels reuse the same lists. Stop at the constituent name.
                block=block.split('% constituentNames(const_index)',1)[0]
                attrs=_attrs(block)
                if variable not in result or attrs:
                    result[variable]={'type':primitive,'dimensions':dimensions,'attrs':attrs}
    return result


@pytest.mark.parametrize('physics',[False,True])
def test_native_generated_primitive_dimensions_and_attributes(physics):
    native=_native_fields(_generated(physics)/'structs_and_variables.inc')
    expected=set(_REGISTRY['fields'] if physics else _REGISTRY['dry_fields'])
    assert set(native)==expected
    for name,definition in native.items():
        ours=field_definition(name,physics=physics)
        assert definition=={k:ours[k] for k in ('type','dimensions','attrs')},name


@pytest.mark.parametrize('physics',[False,True])
def test_native_generated_immutable_membership(physics):
    source=(_generated(physics)/'setup_immutable_streams.inc').read_text()
    streams={};packages=''
    for line in source.splitlines():
        create=re.search(r"create_stream\(manager, '([^']+)'",line)
        if create:streams[create[1]]=[]
        gate=re.search(r"write\(packages,'\(a\)'\) '([^']*)'",line)
        if gate:packages=gate[1]
        member=re.search(r"add_(field|pool)\(manager, '([^']+)', '([^']+)'(.*)\)",line)
        if member:streams[member[2]].append((member[1],member[3],packages if 'packages=packages' in member[4] else ''))
    groups=_REGISTRY['groups' if physics else 'dry_groups']
    # Exercise each package alone as well as all/none. The native emitted
    # member-package strings are independent of the port's stream XML table.
    for active in (frozenset(),_all_packages(),*(frozenset([p]) for p in sorted(_all_packages()))):
        for name,members in streams.items():
            ours=registry_stream(name,physics=physics)
            if not package_active(ours.options.get('packages',''),active):continue
            native=set()
            for kind,member,gate in members:
                if not package_active(gate,active):continue
                expanded=groups.get(member,(member,))
                for variable in expanded:
                    definition=field_definition(variable,physics=physics)
                    if all(package_active(p,active) for p in definition['gates']):native.add(variable)
            assert native==set(ours.fields(active,physics=physics)),(name,active)


def test_generated_default_deck_duplicate_has_native_source_authority():
    from woof.hex.streams_rrfs import parse_streams
    from xml.etree import ElementTree as ET
    root=os.environ.get('RRFS_NATIVE_STREAM_DEFAULTS')
    if not root:pytest.skip('native default stream deck artifact not supplied')
    path=Path(root)/'streams.atmosphere'
    names=[el.attrib['name'] for el in ET.parse(path).getroot()]
    assert names.count('output')==2
    # Original manager create_stream:319-359 rejects an existing name.
    # The callback recorder does not replace that Fortran manager behavior.
    with pytest.raises(ValueError,match='duplicate stream output'):parse_streams(path)


def test_pinned_metadata_regenerates_from_all_source_includes():
    from stream_registry_source_rrfs import metadata
    ref=os.environ.get('RRFS_MPAS_REF')
    if not ref:pytest.skip('pinned MPAS source artifact not supplied')
    assert metadata(Path(ref)/'src/core_atmosphere/Registry.xml')==_REGISTRY
