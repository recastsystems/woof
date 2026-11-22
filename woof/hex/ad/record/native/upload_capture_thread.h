// SPDX-License-Identifier: Apache-2.0
// Hash host uploads by default. Only retain a byte payload when an independently
// certified immutable-input manifest contains its exact SHA-256 and size.
// A retained value is an input candidate, not proof of its origin or lifetime.
static int trace_immutable_upload_allowed(const char *digest, size_t bytes) {
    const char *manifest = getenv("BINDER_IMMUTABLE_UPLOADS_FILE");
    if (!manifest || !*manifest) return 0;
    FILE *file = fopen(manifest, "r");
    if (!file) abort();
    char line[256], hash[65], extra;
    unsigned long long length;
    int allowed = 0;
    while (fgets(line, sizeof line, file)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%64s %llu %c", hash, &length, &extra) != 2 ||
            strlen(hash) != 64 || strspn(hash, "0123456789abcdef") != 64) abort();
        if (!strcmp(hash, digest) && length == bytes) allowed = 1;
    }
    if (ferror(file) || fclose(file)) abort();
    return allowed;
}

static void trace_upload(const void *source, size_t bytes) {
    const char *root = getenv("BINDER_CAPTURE_DIR");
    if (!root) return;
    char path[1024], final[1024], line[640], digest_hex[65];
    unsigned char buffer[65536], digest[SHA256_DIGEST_LENGTH];
    SHA256_CTX hash;
    SHA256_Init(&hash);
    size_t read_bytes = 0;
    while (read_bytes < bytes) {
        size_t chunk = bytes-read_bytes < sizeof buffer ? bytes-read_bytes : sizeof buffer;
        struct iovec local = {buffer, chunk}, remote = {(void *)((uintptr_t)source+read_bytes), chunk};
        ssize_t n = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
        if (n <= 0) break;
        SHA256_Update(&hash, buffer, (size_t)n);
        read_bytes += (size_t)n;
    }
    SHA256_Final(digest, &hash);
    for (size_t i=0; i<sizeof digest; i++) (void)snprintf(digest_hex+2*i, 3, "%02x", digest[i]);
    int eligible = capture_step && read_bytes == bytes && trace_immutable_upload_allowed(digest_hex, bytes);
    /* opt-in retention of small uploads after step 1 (pointer and descriptor
       tables). Their words are device addresses; the sealer resolves them to live allocations and never
       replays them as bytes. Off unless BINDER_RETAIN_SMALL_UPLOADS (max bytes) is set. */
    const char *small = getenv("BINDER_RETAIN_SMALL_UPLOADS");
    if (!eligible && small && capture_step && read_bytes == bytes && bytes <= (size_t)strtoull(small, NULL, 10))
        eligible = 1;
    int retained = 0, retention_error = 0;
    if (eligible) {
        (void)snprintf(path, sizeof path, "%s/payloads/.upload-%ld-%llu.gz", root, (long)getpid(), (unsigned long long)trace_sequence);
        (void)snprintf(final, sizeof final, "%s/payloads/%s.gz", root, digest_hex);
        if (access(final, F_OK) == 0) retained = 1;
        else {
            gzFile out = gzopen(path, "wb1");
            if (!out) retention_error = 1;
            else {
                SHA256_CTX verify;
                unsigned char second[SHA256_DIGEST_LENGTH];
                SHA256_Init(&verify);
                size_t copied = 0;
                while (copied < bytes) {
                    size_t chunk = bytes-copied < sizeof buffer ? bytes-copied : sizeof buffer;
                    struct iovec local = {buffer, chunk}, remote = {(void *)((uintptr_t)source+copied), chunk};
                    ssize_t n = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
                    if (n <= 0 || gzwrite(out, buffer, (unsigned)n) != (int)n) break;
                    SHA256_Update(&verify, buffer, (size_t)n);
                    copied += (size_t)n;
                }
                SHA256_Final(second, &verify);
                int close_status = gzclose(out);
                if (copied != bytes || close_status != Z_OK || memcmp(digest, second, sizeof digest)) {
                    (void)unlink(path);
                    retention_error = 1;
                } else if (rename(path, final)) {
                    (void)unlink(path);
                    retention_error = 1;
                } else retained = 1;
            }
        }
    }
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    int used = snprintf(line, sizeof line,
        "{\"sequence\":%llu,\"clock_ns\":%llu,\"type\":\"HostUploadBytes\",\"source\":%llu,"
        "\"bytes\":%zu,\"written\":%zu,\"sha256\":\"%s\",\"payload\":%s,"
        "\"immutable_manifest_match\":%s,\"retention_error\":%s,\"step\":%d,\"thread_id\":%ld,\"complete\":false}\n",
        (unsigned long long)++trace_sequence,
        (unsigned long long)now.tv_sec*1000000000ULL+(unsigned long long)now.tv_nsec,
        (unsigned long long)(uintptr_t)source, bytes, read_bytes, digest_hex,
        retained ? "true" : "false", eligible ? "true" : "false",
        retention_error ? "true" : "false", capture_step, (long)syscall(SYS_gettid));
    if (used > 0 && (size_t)used < sizeof line) write_line(line, (size_t)used);
}
