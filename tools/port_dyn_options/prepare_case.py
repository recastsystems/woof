"""Make an isolated option case from native real input, without field edits."""
import re
import sys
from pathlib import Path
base, dest = map(Path, sys.argv[1:3])
dest.mkdir(parents=True, exist_ok=True)
for name in ('wrfinput_d01', 'wrfbdy_d01'):
    target=dest/name
    if target.exists():
        assert target.resolve()==(base/name).resolve(), target
    else:
        target.symlink_to(base / name)
text = (base / 'namelist.input').read_text()
for item in sys.argv[3:]:
    key, value = item.split('=', 1)
    pattern = rf'(?im)^\s*{key}\s*=.*$'
    if re.search(pattern, text):
        text = re.sub(pattern, f' {key} = {value},', text)
    else:
        text = text.replace('&dynamics', '&dynamics\n ' + key + ' = ' + value + ',')
(dest / 'namelist.input').write_text(text)
