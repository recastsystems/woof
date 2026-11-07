#!/usr/bin/env python3
"""Extract pinned ROCm user-space packages; never modify the system driver.

Run on Ubuntu 24.04 x86_64 (or inside a user-owned Ubuntu container) with
Python 3, dpkg-deb, g++, libnuma1, libxml2, libtinfo6 and Perl installed.
This installs no kernel module, service, udev rule or system package.
"""
import argparse,concurrent.futures,hashlib,json,os,subprocess,urllib.request
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--downloads',type=Path);p.add_argument('--jobs',type=int,default=4);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 if not 1<=a.jobs<=8:p.error('--jobs must be between 1 and 8')
 prefix=a.prefix.expanduser().resolve()
 if prefix==Path('/'):p.error('use a user-owned prefix')
 download=(a.downloads or prefix/'downloads').expanduser().resolve();download.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(Path(__file__).with_name('rocm-6.4.3-packages.json').read_text())
 def fetch(row):
  path=download/(row['Package']+'.deb')
  if not path.exists():
   temporary=path.with_suffix('.part')
   urllib.request.urlretrieve(manifest['base_url']+row['Filename'],temporary);os.replace(temporary,path)
  actual=hashlib.sha256(path.read_bytes()).hexdigest()
  if actual!=row['SHA256']:raise ValueError('SHA256 mismatch: '+str(path))
  return path
 with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:paths=list(pool.map(fetch,manifest['packages']))
 if not a.verify_only:
  prefix.mkdir(parents=True,exist_ok=True)
  # Shared directories/symlinks require ordered extraction.
  for path in paths:subprocess.run(['dpkg-deb','-x',str(path),str(prefix)],check=True)
  (prefix/'rocm-install-receipt.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'prefix':str(prefix),'verified_packages':len(paths),'extracted':not a.verify_only,'rocm_path':str(prefix/'opt/rocm-6.4.3')},indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
