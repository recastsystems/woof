import os as _os
if _os.environ.get('HEX_MEMCENSUS'):
    import hex_memcensus as _hm
    _hm.install(_os.environ['HEX_MEMCENSUS'])
