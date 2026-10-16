/*
 * arc_render.h - SDL2 + OpenGL 2.1 (fixed function) renderer for one world
 * of a simulation. Used by the interactive viewer, by the "human" render
 * mode, and offscreen (hidden window + framebuffer object) for "rgb_array".
 */
#ifndef ARC_RENDER_H
#define ARC_RENDER_H

#include "arc_sim.h"

typedef struct arc_renderer arc_renderer;

/* visible = 0 renders offscreen. Returns NULL on failure (no display, no
 * OpenGL 2.1). */
arc_renderer *arc_render_create(int width, int height, int visible);
void arc_render_destroy(arc_renderer *r);

/*
 * Draws world `world` of the simulation. markers: n_markers points (x, y, z)
 * drawn as checkpoints, the one at index `active` highlighted (-1: none).
 * If rgb is not NULL, it receives width * height * 3 bytes, top row first.
 * Returns 0 once the window has been closed (Escape / Q / close button).
 */
int arc_render_frame(arc_renderer *r, const arc_sim *s, int world,
                     const double *markers, int n_markers, int active,
                     unsigned char *rgb);

/* Size of the frames returned by arc_render_frame. */
void arc_render_size(const arc_renderer *r, int *width, int *height);

/* Keys pressed since the previous call (SDL keycodes, at most max), for
 * applications reacting to input. Returns the count. */
int arc_render_keys(arc_renderer *r, int *keys, int max);

void arc_render_set_title(arc_renderer *r, const char *title);

#endif
