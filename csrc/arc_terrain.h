/*
 * arc_terrain.h - random block terrain.
 */
#ifndef ARC_TERRAIN_H
#define ARC_TERRAIN_H

#include "arc_phys.h"
#include "arc_rng.h"

/*
 * Fills t with an n x n grid of columns centered on the origin: smooth hills
 * plus per-column jitter, quantized into steps. difficulty in [0, 1] scales
 * every height (0 gives flat ground). Heights are shifted so that the
 * column at the origin is at z = 0.
 */
void arc_terrain_generate(arc_terrain *t, int n, double cell, double difficulty, arc_rng *rng);

#endif
