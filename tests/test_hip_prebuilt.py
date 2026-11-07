import hashlib,json,zipfile
import pytest
from woof import hip_prebuilt as p

def test_hip_header_binding_ignores_bytecode_but_detects_header_changes(tmp_path):
    from woof.kernel_dependencies import include_tree_dependencies
    header=tmp_path/'kernel.cuh';header.write_text('#define VALUE 1\n')
    options=('-I'+str(tmp_path),)
    def hip():
        return include_tree_dependencies(options,{},exclude_python_bytecode=True)
    original=hip()
    generated=tmp_path/'__pycache__';generated.mkdir()
    (generated/'kernels.cpython-314.pyc').write_bytes(b'interpreter-specific')
    assert hip()==original
    assert include_tree_dependencies(options,{})!=original
    header.write_text('#define VALUE 2\n')
    assert hip()!=original

def test_relocation_preserves_content_binding_and_rejects_changes(tmp_path,monkeypatch):
    blob=b'\x7fELFtest';digest=hashlib.sha256(blob).hexdigest()
    value=p.binding('source','translation','gfx942',('-std=c++17','-I/old/header'),
                    [{'compiler':'digest'},{'tree_sha256':'headers'}],'translator')
    moved=p.binding('source','translation','gfx942',('-std=c++17','-I/new/header'),
                    [{'compiler':'digest'},{'tree_sha256':'headers'}],'translator')
    assert moved==value
    path=tmp_path/'objects.zip';member='images/'+digest+'.hsaco'
    receipt=dict(key='build-key',image_sha256=digest)
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(member,blob)
        z.writestr('MANIFEST.json',json.dumps(dict(schema=p.SCHEMA,objects={p.key(value):dict(binding=value,image=member,receipt=receipt)})))
    monkeypatch.setenv('WOOF_HIP_PREBUILT',str(path))
    assert p.lookup(moved)[0]==blob
    for field,replacement in [('source_sha256','other'),('translated_sha256','other'),
        ('arch','gfx1100'),('headers',[{'tree_sha256':'changed'}]),('options',['-ffast-math'])]:
        assert p.lookup(dict(moved,**{field:replacement})) is None
    with zipfile.ZipFile(path,'a') as z:z.writestr(member,b'\x7fELFbroken')
    with pytest.raises(ValueError,match='Corrupt'):p.lookup(moved)
