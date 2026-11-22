// SPDX-License-Identifier: Apache-2.0
// Card-to-host capture: the bytes one host read received, kept in payloads/ under their SHA-256 name.
//
// THE BREAKAGE THIS PREVENTS (x1.40962 door recording from a start file, 2026-11-11): the capture hashed the
// host buffer of the caller and then wrote that same live buffer in a second pass, and for an asynchronous copy
// given no stream it did not wait for the copy to land. 13 reads of 163,848 bytes were hashed while their buffer
// was still zero, and the file named for 163,848 zero bytes was written after other bytes had arrived (12 fives
// and 40,950 sixes). 186 later uploads of real zeros found a file of that name and kept it, so the sealed plan
// refused its own replay with "SHA-256 mismatch".
//
// Two rules, both here so they can be run without a card (tests/test_ad_record_cpu.py):
//   d2h_wait   an asynchronous copy is waited for before its bytes are read: on its stream when it has one, on
//              the whole context when it has none. If it cannot be waited for, nothing is hashed or kept.
//   d2h_store  ONE private copy of the host bytes is taken; that copy is hashed and that copy is written, so a
//              file name cannot disagree with its content whatever happens to the buffer meanwhile.
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>
#include <openssl/sha.h>

typedef int (*d2h_sync_stream_fn)(void *);
typedef int (*d2h_sync_context_fn)(void);

/* 0 when the host bytes of this copy may be read; else the driver status (or -1: no way to wait). */
static int d2h_wait(int asynchronous, void *stream, d2h_sync_stream_fn sync_stream, d2h_sync_context_fn sync_context) {
    if (!asynchronous) return 0;
    if (stream) return sync_stream ? sync_stream(stream) : -1;
    return sync_context ? sync_context() : -1;
}

/* Keeps `bytes` bytes at `host` as ROOT/payloads/<sha256>.gz and gives the name in hex. 0, or a negative step. */
static int d2h_store(const char *root, const void *host, size_t bytes, unsigned long long serial, char hex[65]) {
    unsigned char *copy = malloc(bytes ? bytes : 1);
    if (!copy) return -1;
    memcpy(copy, host, bytes);
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(copy, bytes, digest);
    for (size_t j = 0; j < sizeof digest; j++) (void)snprintf(hex + 2 * j, 3, "%02x", digest[j]);
    char path[1100], final[1100];
    (void)snprintf(final, sizeof final, "%s/payloads/%s.gz", root, hex);
    int rc = 0;
    if (access(final, F_OK)) {
        (void)snprintf(path, sizeof path, "%s/payloads/.d2h-%ld-%llu.gz", root, (long)getpid(), serial);
        gzFile out = gzopen(path, "wb1");
        if (!out) rc = -2;
        else {
            if (gzwrite(out, copy, (unsigned)bytes) != (int)bytes) rc = -3;
            if (gzclose(out) != Z_OK && !rc) rc = -4;
            if (!rc && link(path, final) && errno != EEXIST) rc = -5;
            (void)unlink(path);
        }
    }
    free(copy);
    return rc;
}
