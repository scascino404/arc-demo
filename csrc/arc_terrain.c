/*
 * arc_terrain.c - random block terrain: two octaves of value noise for the
 * hills, uniform per-column jitter for the blocky look, then quantization so
 * neighbouring columns form distinct steps.
 */
#include "arc_terrain.h"

#define HILL_AMPLITUDE 0.45     /* m, at difficulty 1 */
#define HILL_SPACING 8          /* cells between lattice points */
#define BUMP_AMPLITUDE 0.12
#define BUMP_SPACING 3
#define JITTER_AMPLITUDE 0.06
#define STEP 0.03               /* quantization of column heights */
#define WALL 1.0                /* boundary wall height above the highest column */

#define LATTICE_MAX (ARC_GRID_MAX / BUMP_SPACING + 3)

static double smooth(double t) { return t * t * (3.0 - 2.0 * t); }

/* Adds amplitude * value noise with the given lattice spacing (in cells). */
static void add_noise(arc_terrain *t, int spacing, double amplitude, arc_rng *rng)
{
    double local[LATTICE_MAX * LATTICE_MAX];
    int m = t->nx / spacing + 2, ix, iy;
    for (ix = 0; ix < m * m; ++ix)
        local[ix] = arc_rng_range(rng, -1.0, 1.0);
    for (iy = 0; iy < t->ny; ++iy) {
        for (ix = 0; ix < t->nx; ++ix) {
            double fx = (double)ix / spacing, fy = (double)iy / spacing;
            int gx = (int)fx, gy = (int)fy;
            double tx = smooth(fx - gx), ty = smooth(fy - gy);
            double a = local[gx + gy * m], b = local[gx + 1 + gy * m];
            double c = local[gx + (gy + 1) * m], d = local[gx + 1 + (gy + 1) * m];
            double v = (a + (b - a) * tx) * (1.0 - ty) + (c + (d - c) * tx) * ty;
            t->h[ix + iy * t->nx] += amplitude * v;
        }
    }
}

void arc_terrain_generate(arc_terrain *t, int n, double cell, double difficulty, arc_rng *rng)
{
    int i;
    double base, top;
    arc_terrain_flat(t, n, n, cell, 0.0);
    t->outside = WALL;
    if (difficulty <= 0.0)
        return;
    add_noise(t, HILL_SPACING, HILL_AMPLITUDE * difficulty, rng);
    add_noise(t, BUMP_SPACING, BUMP_AMPLITUDE * difficulty, rng);
    for (i = 0; i < t->nx * t->ny; ++i) {
        double h = t->h[i] + JITTER_AMPLITUDE * difficulty * arc_rng_range(rng, -1.0, 1.0);
        t->h[i] = STEP * floor(h / STEP + 0.5);
    }
    base = arc_terrain_height(t, 0.0, 0.0);
    top = -1e30;
    for (i = 0; i < t->nx * t->ny; ++i) {
        t->h[i] -= base;
        top = arc_max(top, t->h[i]);
    }
    t->outside = top + WALL;
}
