/*
 * arc_rng.h - small deterministic RNG (xorshift128, C89: 32-bit arithmetic
 * on unsigned long, masked so it behaves the same on 32 and 64 bit longs).
 */
#ifndef ARC_RNG_H
#define ARC_RNG_H

#include "arc_math.h"

#define ARC_U32(v) ((v) & 0xffffffffUL)

typedef struct { unsigned long s[4]; } arc_rng;

ARC_INLINE unsigned long arc_rng_u32(arc_rng *r)
{
    unsigned long t = r->s[3], s = r->s[0];
    r->s[3] = r->s[2];
    r->s[2] = r->s[1];
    r->s[1] = s;
    t = ARC_U32(t ^ (t << 11));
    t ^= t >> 8;
    r->s[0] = ARC_U32(t ^ s ^ (s >> 19));
    return r->s[0];
}

/* splitmix-style scrambling so that nearby seeds give unrelated streams */
ARC_INLINE unsigned long arc_hash32(unsigned long x)
{
    x = ARC_U32(x);
    x = ARC_U32((x ^ (x >> 16)) * 0x7feb352dUL);
    x = ARC_U32((x ^ (x >> 15)) * 0x846ca68bUL);
    return ARC_U32(x ^ (x >> 16));
}

ARC_INLINE void arc_rng_seed(arc_rng *r, unsigned long seed)
{
    int i;
    for (i = 0; i < 4; ++i) {
        seed = arc_hash32(seed + 0x9e3779b9UL * (i + 1));
        r->s[i] = seed ? seed : 1UL;
    }
}

/* uniform in [0, 1) */
ARC_INLINE double arc_rng_uniform(arc_rng *r)
{
    return arc_rng_u32(r) * (1.0 / 4294967296.0);
}

ARC_INLINE double arc_rng_range(arc_rng *r, double lo, double hi)
{
    return lo + (hi - lo) * arc_rng_uniform(r);
}

#endif
