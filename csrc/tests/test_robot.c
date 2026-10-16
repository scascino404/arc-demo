/*
 * test_robot.c - quadruped and batched simulation checks: standing still on
 * flat ground, surviving random actions on rough terrain, determinism and
 * thread independence.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../arc_sim.h"
#include "../arc_rng.h"

#define DT (1.0 / 30.0)
#define SUBSTEPS 20
#define NJ ARC_ROBOT_JOINTS

static int n_fail = 0, n_pass = 0;

static void check(int ok, const char *name, double got, double expected)
{
    if (ok) ++n_pass; else ++n_fail;
    printf("  %-5s %-44s got % .6f (expected % .6f)\n", ok ? "ok" : "FAIL", name, got, expected);
}

static void check_below(const char *name, double got, double limit)
{
    check(got <= limit, name, got, limit);
}

static void reset_all(arc_sim *s, unsigned int seed, double difficulty)
{
    int n = arc_sim_num_worlds(s), i;
    int *ids = (int *)malloc(sizeof(int) * n);
    unsigned int *seeds = (unsigned int *)malloc(sizeof(unsigned int) * n);
    double *diff = (double *)malloc(sizeof(double) * n);
    for (i = 0; i < n; ++i) {
        ids[i] = i;
        seeds[i] = seed + i;
        diff[i] = difficulty;
    }
    arc_sim_reset(s, ids, n, seeds, diff);
    free(ids);
    free(seeds);
    free(diff);
}

static void test_stand(void)
{
    arc_sim *s = arc_sim_create(1, DT, SUBSTEPS, 32, 0.4);
    double def[NJ], lo[NJ], hi[NJ], misc[2], targets[NJ], root[ARC_SIM_ROOT_SIZE];
    double contacts[ARC_SIM_CONTACT_SIZE], maxv = 0.0;
    int i, flags;
    printf("standing on flat ground\n");
    arc_sim_robot_info(s, def, lo, hi, misc);
    printf("  stand height %.3f m, mass %.1f kg\n", misc[0], misc[1]);
    reset_all(s, 1, 0.0);
    memcpy(targets, def, sizeof(def));
    for (i = 0; i < 150; ++i) {
        arc_sim_step(s, targets);
        arc_sim_observe(s, root, 0, contacts, &flags);
        if (i >= 90)
            maxv = arc_max(maxv, sqrt(root[7] * root[7] + root[8] * root[8] + root[9] * root[9]));
    }
    check(!flags, "stable", flags, 0);
    check(root[2] > 0.8 * misc[0], "torso stays up (m)", root[2], 0.8 * misc[0]);
    check_below("torso speed after settling (m/s)", maxv, 0.01);
    check(contacts[0] > 0.99 && contacts[1] > 0.99 && contacts[2] > 0.99 && contacts[3] > 0.99,
          "all feet in contact", contacts[0], 1.0);
    check_below("no torso contact", contacts[ARC_ROBOT_LEGS], 0.0);
    /* upright: body z axis close to world z */
    check(1.0 - 2.0 * (root[4] * root[4] + root[5] * root[5]) > 0.99, "upright",
          1.0 - 2.0 * (root[4] * root[4] + root[5] * root[5]), 0.99);
    arc_sim_destroy(s);
}

/* Random smooth-ish actions on hard terrain: nothing may blow up. */
static void run_random(arc_sim *s, unsigned int seed, int steps, double *maxspeed, int *unstable)
{
    int n = arc_sim_num_worlds(s), i, k, j;
    double def[NJ], lo[NJ], hi[NJ];
    double *targets = (double *)malloc(sizeof(double) * n * NJ);
    double *root = (double *)malloc(sizeof(double) * n * ARC_SIM_ROOT_SIZE);
    int *flags = (int *)malloc(sizeof(int) * n);
    arc_rng rng;
    arc_rng_seed(&rng, seed);
    arc_sim_robot_info(s, def, lo, hi, 0);
    *maxspeed = 0.0;
    *unstable = 0;
    for (i = 0; i < n * NJ; ++i)
        targets[i] = def[i % NJ];
    for (k = 0; k < steps; ++k) {
        /* new random targets every 3 steps (10 Hz), full joint range */
        if (k % 3 == 0)
            for (i = 0; i < n * NJ; ++i)
                targets[i] = arc_rng_range(&rng, lo[i % NJ], hi[i % NJ]);
        arc_sim_step(s, targets);
        arc_sim_observe(s, root, 0, 0, flags);
        for (j = 0; j < n; ++j) {
            double *r = root + j * ARC_SIM_ROOT_SIZE;
            *maxspeed = arc_max(*maxspeed, sqrt(r[7] * r[7] + r[8] * r[8] + r[9] * r[9]));
            *unstable += flags[j];
        }
    }
    free(targets);
    free(root);
    free(flags);
}

static void test_random(void)
{
    arc_sim *s = arc_sim_create(16, DT, SUBSTEPS, 48, 0.4);
    double maxspeed;
    int unstable;
    printf("random actions, 16 worlds x 20 s, difficulty 1\n");
    arc_sim_set_threads(s, 4);
    reset_all(s, 100, 1.0);
    run_random(s, 7, 600, &maxspeed, &unstable);
    check(unstable == 0, "no unstable worlds", unstable, 0);
    check_below("max torso speed (m/s)", maxspeed, 6.0);
    arc_sim_destroy(s);
}

static void test_determinism(void)
{
    arc_sim *a = arc_sim_create(8, DT, SUBSTEPS, 48, 0.4);
    arc_sim *b = arc_sim_create(8, DT, SUBSTEPS, 48, 0.4);
    double ra[8 * ARC_SIM_ROOT_SIZE], rb[8 * ARC_SIM_ROOT_SIZE], ms;
    int u;
    printf("determinism (1 vs 8 threads)\n");
    arc_sim_set_threads(a, 1);
    arc_sim_set_threads(b, 8);
    reset_all(a, 5, 0.7);
    reset_all(b, 5, 0.7);
    run_random(a, 3, 120, &ms, &u);
    run_random(b, 3, 120, &ms, &u);
    arc_sim_observe(a, ra, 0, 0, 0);
    arc_sim_observe(b, rb, 0, 0, 0);
    check(memcmp(ra, rb, sizeof(ra)) == 0, "bitwise identical states", 0, 0);
    arc_sim_destroy(a);
    arc_sim_destroy(b);
}

int main(void)
{
    test_stand();
    test_random();
    test_determinism();
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
