#!/usr/bin/env python3
"""Transparent compiler wrapper: append actual cwd/argv/status after compilation."""
import json
import os
from pathlib import Path
import subprocess
import sys

name = Path(sys.argv[0]).name
actual = json.loads(os.environ['RRFS_REAL_COMPILERS'])[name]
args = sys.argv[1:]
status = subprocess.run([actual, *args]).returncode
if '-c' in args:
    record = json.dumps(dict(cwd=os.getcwd(), compiler=name, argv=args, status=status)) + '\n'
    fd = os.open(os.environ['RRFS_COMPILE_RECEIPT'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, record.encode())
    finally:
        os.close(fd)
sys.exit(status)
