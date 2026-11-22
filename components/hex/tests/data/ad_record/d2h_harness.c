// SPDX-License-Identifier: Apache-2.0
// CPU harness for src/hexcore/ad/record/native/d2h_capture.h (tests/test_ad_record_cpu.py). No card, no driver.
//   harness store ROOT N BYTES   a second thread rewrites the buffer all the time while N captures are stored.
//   harness wait ROOT            an "asynchronous copy" lands only when the capture waits; prints the kept name.
// Built with -DOLD_CAPTURE it runs the capture as it was before the fix (the control: it must go wrong).
#include <pthread.h>
#include <stdatomic.h>
#ifdef OLD_CAPTURE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>
#include <openssl/sha.h>
static int d2h_store(const char *root, const void *host, size_t bytes, unsigned long long serial, char hex[65]) {
    (void)serial;
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256((const unsigned char *)host, bytes, digest);
    for (size_t j = 0; j < sizeof digest; j++) (void)snprintf(hex + 2 * j, 3, "%02x", digest[j]);
    char path[1024], final[1024];
    (void)snprintf(final, sizeof final, "%s/payloads/%s.gz", root, hex);
    if (access(final, F_OK)) {
        (void)snprintf(path, sizeof path, "%s/payloads/.d2h-%ld.gz", root, (long)getpid());
        gzFile out = gzopen(path, "wb1");
        int ok = out && gzwrite(out, host, (unsigned)bytes) == (int)bytes;
        if (out && gzclose(out) != Z_OK) ok = 0;
        if (ok && link(path, final) && errno != EEXIST) ok = 0;
        unlink(path);
        if (!ok) return -1;
    }
    return 0;
}
typedef int (*d2h_sync_stream_fn)(void *);
typedef int (*d2h_sync_context_fn)(void);
static int d2h_wait(int asynchronous, void *stream, d2h_sync_stream_fn a, d2h_sync_context_fn b) {
    (void)b;
    if (asynchronous && stream) return a ? a(stream) : -1;   /* no stream: no wait (the defect) */
    return 0;
}
#else
#include "d2h_capture.h"
#endif

static unsigned char *buffer;
static size_t size;
static atomic_int stop;
static void *mutate(void *unused) {
    (void)unused;
    for (unsigned v = 1; !atomic_load(&stop); v++) memset(buffer, (int)(v & 255), size);
    return NULL;
}
static unsigned char pending[4096];
static int stream_waits, context_waits;
static int fake_stream(void *s) { (void)s; stream_waits++; memcpy(buffer, pending, sizeof pending); return 0; }
static int fake_context(void) { context_waits++; memcpy(buffer, pending, sizeof pending); return 0; }

int main(int argc, char **argv) {
    char hex[65];
    if (argc == 5 && !strcmp(argv[1], "store")) {
        long n = strtol(argv[3], NULL, 10);
        size = (size_t)strtoull(argv[4], NULL, 10);
        buffer = calloc(size, 1);
        pthread_t th;
        pthread_create(&th, NULL, mutate, NULL);
        int bad = 0;
        for (long i = 0; i < n; i++) if (d2h_store(argv[2], buffer, size, (unsigned long long)i + 1, hex)) bad++;
        atomic_store(&stop, 1);
        pthread_join(th, NULL);
        printf("{\"stored\":%ld,\"errors\":%d}\n", n, bad);
        return bad ? 1 : 0;
    }
    if (argc == 3 && !strcmp(argv[1], "wait")) {
        size = sizeof pending;
        buffer = calloc(size, 1);
        for (size_t i = 0; i < size; i += 4) pending[i] = (unsigned char)(i / 4 < 12 ? 5 : 6);
        /* 1: asynchronous, no stream (the recorded case: MemcpyDtoHAsync on stream 0) */
        int rc = d2h_wait(1, NULL, fake_stream, fake_context);
        if (rc || d2h_store(argv[2], buffer, size, 1, hex)) return 2;
        printf("{\"case\":\"async-no-stream\",\"sha256\":\"%s\",\"stream_waits\":%d,\"context_waits\":%d}\n", hex, stream_waits, context_waits);
        /* 2: asynchronous on a stream */
        memset(buffer, 0, size);
        stream_waits = context_waits = 0;
        if (d2h_wait(1, (void *)buffer, fake_stream, fake_context)) return 4;
        printf("{\"case\":\"async-stream\",\"stream_waits\":%d,\"context_waits\":%d}\n", stream_waits, context_waits);
        /* 3: synchronous: no wait is made */
        stream_waits = context_waits = 0;
        if (d2h_wait(0, NULL, fake_stream, fake_context)) return 5;
        printf("{\"case\":\"sync\",\"stream_waits\":%d,\"context_waits\":%d}\n", stream_waits, context_waits);
        /* 4: asynchronous and no way to wait: refused, so nothing is hashed */
        printf("{\"case\":\"async-no-driver\",\"rc\":%d}\n", d2h_wait(1, NULL, NULL, NULL));
        return 0;
    }
    return 64;
}
