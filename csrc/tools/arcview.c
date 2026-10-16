/*
 * arcview.c - interactive physics viewer, no learning involved: the robot
 * either holds its default stance or follows random joint targets.
 *
 * Keys: Space hold / random actions, R new terrain, 0-9 terrain difficulty,
 *       P pause, N single step (paused), S slow motion, C camera view,
 *       Esc/Q quit.
 * Mouse: left drag orbits the camera, wheel zooms.
 *
 * Offscreen screenshot (for automated checks):
 *   arcview --shot out.ppm [--steps N] [--difficulty D] [--random] [--seed S]
 *           [--view 0|1|2]   (camera behind / side / front)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL.h>
#include "../arc_render.h"
#include "../arc_rng.h"

#define NJ ARC_ROBOT_JOINTS
#define DT (1.0 / 30.0)

typedef struct {
    arc_sim *sim;
    double def[NJ], lo[NJ], hi[NJ], targets[NJ];
    double difficulty;
    unsigned int seed;
    int random_actions;
    arc_rng rng;
    int step;
} viewer;

static void reset(viewer *v)
{
    int id = 0;
    arc_sim_reset(v->sim, &id, 1, &v->seed, &v->difficulty);
    memcpy(v->targets, v->def, sizeof(v->def));
    v->step = 0;
}

static void step(viewer *v)
{
    int i;
    if (v->random_actions && v->step % 3 == 0)
        for (i = 0; i < NJ; ++i)
            v->targets[i] = arc_rng_range(&v->rng, v->lo[i], v->hi[i]);
    if (!v->random_actions)
        memcpy(v->targets, v->def, sizeof(v->def));
    arc_sim_step(v->sim, v->targets);
    ++v->step;
}

static int write_ppm(const char *path, const unsigned char *rgb, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, (size_t)w * h * 3, f);
    fclose(f);
    return 1;
}

static int screenshot(viewer *v, const char *path, int steps, int view)
{
    arc_renderer *r = arc_render_create(960, 540, 0);
    unsigned char *rgb;
    int i, w, h;
    if (!r) {
        fprintf(stderr, "arcview: cannot create offscreen renderer: %s\n", SDL_GetError());
        return 1;
    }
    arc_render_set_view(r, view);
    arc_render_size(r, &w, &h);
    rgb = (unsigned char *)malloc((size_t)w * h * 3);
    for (i = 0; i < steps; ++i) {
        step(v);
        arc_render_frame(r, v->sim, 0, 0, 0, -1, 0); /* lets the camera follow */
    }
    arc_render_frame(r, v->sim, 0, 0, 0, -1, rgb);
    if (!write_ppm(path, rgb, w, h))
        fprintf(stderr, "arcview: cannot write %s\n", path);
    free(rgb);
    arc_render_destroy(r);
    return 0;
}

int main(int argc, char **argv)
{
    viewer v;
    arc_renderer *r;
    const char *shot = 0;
    int i, paused = 0, slow = 0, steps = 90, view = 0, keys[32], nk;
    Uint32 next_tick;
    char title[160];

    memset(&v, 0, sizeof(v));
    v.difficulty = 0.5;
    v.seed = 1;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--shot") && i + 1 < argc) shot = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--difficulty") && i + 1 < argc) v.difficulty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) v.seed = (unsigned int)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--random")) v.random_actions = 1;
        else if (!strcmp(argv[i], "--view") && i + 1 < argc) view = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: arcview [--shot out.ppm] [--steps N] [--difficulty D] [--seed S] [--random] [--view V]\n");
            return 2;
        }
    }

    v.sim = arc_sim_create(1, DT, 20, 48, 0.4);
    arc_sim_robot_info(v.sim, v.def, v.lo, v.hi, 0);
    arc_rng_seed(&v.rng, v.seed);
    reset(&v);
    if (shot) {
        int rc = screenshot(&v, shot, steps, view);
        arc_sim_destroy(v.sim);
        return rc;
    }

    r = arc_render_create(1280, 720, 1);
    if (!r) {
        fprintf(stderr, "arcview: cannot create window: %s\n", SDL_GetError());
        return 1;
    }
    next_tick = SDL_GetTicks();
    for (;;) {
        nk = arc_render_keys(r, keys, 32);
        for (i = 0; i < nk; ++i) {
            int k = keys[i];
            if (k == SDLK_SPACE) v.random_actions = !v.random_actions;
            else if (k == SDLK_r) { ++v.seed; reset(&v); }
            else if (k >= SDLK_0 && k <= SDLK_9) { v.difficulty = (k - SDLK_0) / 9.0; reset(&v); }
            else if (k == SDLK_p) paused = !paused;
            else if (k == SDLK_n && paused) step(&v);
            else if (k == SDLK_s) slow = !slow;
        }
        if (!paused)
            step(&v);
        sprintf(title, "arcview - %s%s%s - difficulty %.2f - seed %u - t %.1f s%s",
                v.random_actions ? "random actions" : "holding stance",
                paused ? " (paused)" : "", slow ? " (slow)" : "", v.difficulty, v.seed,
                v.step * DT, arc_sim_world(v.sim, 0)->unstable ? " - UNSTABLE" : "");
        arc_render_set_title(r, title);
        if (!arc_render_frame(r, v.sim, 0, 0, 0, -1, 0))
            break;
        /* real time pacing (vsync may already throttle) */
        next_tick += (Uint32)(1000.0 * DT * (slow ? 4.0 : 1.0));
        {
            Uint32 now = SDL_GetTicks();
            if (next_tick > now)
                SDL_Delay(next_tick - now);
            else
                next_tick = now;
        }
    }
    arc_render_destroy(r);
    arc_sim_destroy(v.sim);
    return 0;
}
