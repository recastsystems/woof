"""Word-for-word comparison of two POST files (same layout); prints a JSON row."""
import hashlib, json, sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype='<u4'); b = np.fromfile(sys.argv[2], dtype='<u4')
assert a.size == b.size, (a.size, b.size)
d = int(np.count_nonzero(a != b))
print(json.dumps({'a': sys.argv[1], 'b': sys.argv[2], 'words': int(a.size), 'different': d,
                  'a_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'b_sha256': hashlib.sha256(b.tobytes()).hexdigest()}))
