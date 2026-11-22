#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S03: seed a recording's content-addressed immutable-upload payloads from earlier recordings (hard links).

THE COST THIS REMOVES (S03, RTX 5090, 2026-11-07, py-spy on a 3-step pkg18m2-recipe recording): the
recorder (driver_trace_cupti_held.c trace_upload) retains every upload listed in BINDER_IMMUTABLE_UPLOADS_FILE as
payloads/<sha256>.gz and gzip-compresses it on the calling thread, i.e. the forecast's main thread: about 1 GB
per recording (the TEMPO session tables, five 128 MB uploads, then 69 MB and smaller), 42 % of the main thread's
samples, nearly all of the first step's 51 s at one core with the card idle. The recorder already skips the
compression when payloads/<sha256>.gz exists ("retained", the same trace record). These files are named by the
SHA-256 of their content and were written and verified (a second SHA-256 pass) by the same recorder, so a link to
an earlier recording's file is the file this recording would write.

What it does, before the forecast starts:
1. refresh STORE (default RUN/payload-store): every manifest hash found in RUN/*-cupti-recorder/payloads and not
   yet in the store is hard-linked into it (copied when the link crosses filesystems);
2. link every manifest hash the store holds into CAPTURE_PAYLOADS.
Only manifest hashes are seeded (the uploads the recorder retains as immutable inputs). Usage:
seed_payloads.py MANIFEST STORE RUN CAPTURE_PAYLOADS
"""
import errno
import glob
import os
import shutil
import sys
import time


def link(src, dst):
    try:
        os.link(src, dst)
    except FileExistsError:
        return 0
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        tmp = dst + ".seed-tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    return 1


def main():
    manifest, store, run, target = sys.argv[1:5]
    t0 = time.monotonic()
    hashes = []
    with open(manifest) as f:
        for line in f:
            words = line.split()
            if words and not words[0].startswith("#") and len(words[0]) == 64:
                hashes.append(words[0])
    os.makedirs(store, exist_ok=True)
    os.makedirs(target, exist_ok=True)
    sources = [d for d in glob.glob(os.path.join(run, "*-cupti-recorder", "payloads"))
               if os.path.realpath(d) != os.path.realpath(target)]
    added = seeded = 0
    for h in hashes:
        name = h + ".gz"
        kept = os.path.join(store, name)
        if not os.path.exists(kept):
            for d in sources:
                candidate = os.path.join(d, name)
                if os.path.isfile(candidate):
                    added += link(candidate, kept)
                    break
        if os.path.exists(kept):
            seeded += link(kept, os.path.join(target, name))
    print(f"seed_payloads: manifest {len(hashes)}, store +{added}, seeded {seeded} into {target} "
          f"({time.monotonic() - t0:.2f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
