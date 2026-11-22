// SPDX-License-Identifier: Apache-2.0
// Serial SHA-256 for observing a device buffer without copying that buffer.
// The 32-byte digest is the only device-to-host transfer introduced here.
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 ror(u32 x, int n) {
    return (x >> n) | (x << (32 - n));
}

extern "C" __global__ void binder_sha256(const unsigned char *src, u64 len,
                                         unsigned char *digest) {
    if (blockIdx.x || threadIdx.x) return;
    const u32 k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    u32 h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u64 blocks = len / 64 + ((len % 64) < 56 ? 1 : 2);
    u64 padded = blocks * 64;
    u64 bit_len = len * 8;
    for (u64 block = 0; block < blocks; ++block) {
        u32 w[64];
        for (int i = 0; i < 16; ++i) {
            u32 v = 0;
            for (int j = 0; j < 4; ++j) {
                u64 index = block * 64 + i * 4 + j;
                unsigned char byte = 0;
                if (index < len) byte = src[index];
                else if (index == len) byte = 0x80;
                else if (index >= padded - 8)
                    byte = (unsigned char)(bit_len >> (8 * (padded - 1 - index)));
                v = (v << 8) | byte;
            }
            w[i] = v;
        }
        for (int i = 16; i < 64; ++i) {
            u32 x = w[i-15], y = w[i-2];
            w[i] = w[i-16] + (ror(x,7)^ror(x,18)^(x>>3)) + w[i-7]
                   + (ror(y,17)^ror(y,19)^(y>>10));
        }
        u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],q=h[7];
        for (int i = 0; i < 64; ++i) {
            u32 t1=q+(ror(e,6)^ror(e,11)^ror(e,25))+((e&f)^((~e)&g))+k[i]+w[i];
            u32 t2=(ror(a,2)^ror(a,13)^ror(a,22))+((a&b)^(a&c)^(b&c));
            q=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=q;
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j)
            digest[4*i+j] = (unsigned char)(h[i] >> (24-8*j));
}
