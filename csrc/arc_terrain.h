/*
 * arc_terrain.h - block terrain: a grid of axis-aligned columns, and a
 * random generator for it.
 */
#ifndef ARC_TERRAIN_H
#define ARC_TERRAIN_H

#include "arc_rng.h"

#define ARC_GRID_MAX 64
#define ARC_TERRAIN_VOID -100.0 /* height outside generated terrain: a void to fall into */

typedef struct {
    int nx, ny;
    double cell;       /* column width (m) */
    double ox, oy;     /* world position of the minimum corner of cell (0, 0) */
    double outside;    /* height of the columns outside the grid */
    double h[ARC_GRID_MAX * ARC_GRID_MAX]; /* column heights, h[ix + iy * nx] */
} arc_terrain;

/* nx x ny columns of the given height, centered on the origin. */
void arc_terrain_flat(arc_terrain *t, int nx, int ny, double cell, double height);

/*
 * Fills t with an n x n grid of columns centered on the origin: smooth hills
 * plus per-column jitter, quantized into steps. difficulty in [0, 1] scales
 * every height (0 gives flat ground). Heights are shifted so that the
 * column at the origin is at z = 0. Outside the grid is a void: columns
 * at ARC_TERRAIN_VOID, far below.
 */
void arc_terrain_generate(arc_terrain *t, int n, double cell, double difficulty, arc_rng *rng);

/* Height of column (ix, iy); t->outside beyond the grid. */
ARC_INLINE double arc_terrain_cell(const arc_terrain *t, int ix, int iy)
{
    if (ix < 0 || iy < 0 || ix >= t->nx || iy >= t->ny)
        return t->outside;
    return t->h[ix + iy * t->nx];
}

/* Height of the column containing the point (x, y). */
ARC_INLINE double arc_terrain_height(const arc_terrain *t, double x, double y)
{
    /* range-checked as doubles: converting a far-off or NaN coordinate to
     * int is undefined */
    double fx = floor((x - t->ox) / t->cell), fy = floor((y - t->oy) / t->cell);
    if (!(fx >= 0.0 && fy >= 0.0 && fx < t->nx && fy < t->ny))
        return t->outside;
    return t->h[(int)fx + (int)fy * t->nx];
}

#endif
