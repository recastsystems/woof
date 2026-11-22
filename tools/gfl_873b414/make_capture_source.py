"""Write the capture copy of WRF v4.8.0's GFL driver: the original routine renamed, a wrapper added.

usage: make_capture_source.py ORIGINAL_F90 WRAPPER_F90 OUT_F
The original routine body is untouched; only its name changes (subroutine and end line).
"""
import re
import sys
from pathlib import Path

src, wrapper, out = map(Path, sys.argv[1:])
text = src.read_text()
head = ' subroutine cu_grell_freitas_li('
tail = ' end subroutine cu_grell_freitas_li'
assert text.count(head) == 1 and text.count(tail) == 1, 'unexpected driver layout'
text = text.replace(head, ' subroutine gfl_referee_body(').replace(tail, ' end subroutine gfl_referee_body')
end = re.search(r'^\s*END\s+MODULE\s+module_cu_gfl\b.*$', text, re.I | re.M)
assert end, 'module end not found'
text = text[:end.start()] + '\n' + wrapper.read_text() + '\n' + text[end.start():]
out.write_text(text)
print('capture source written', out)
