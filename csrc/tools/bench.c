/*
 * bench.c - raw simulation throughput (control steps per second) for a
 * range of thread counts, with random actions on rough terrain.
 *   usage: bench [worlds] [seconds-per-config]
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../arc_sim.h"
#include "../arc_rng.h"

#define NJ ARC_ROBOT_JOINTS

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 64;
    double secs = argc > 2 ? atof(argv[2]) : 2.0;
    static const int threads[] = { 1, 2, 4, 8, 16 };
    arc_sim *s = arc_sim_create(n, 1.0 / 30.0, 20, 48, 0.4);
    double lo[NJ], hi[NJ], *targets = (double *)malloc(sizeof(double) * n * NJ);
    int *ids = (int *)malloc(sizeof(int) * n), i, t;
    unsigned int *seeds = (unsigned int *)malloc(sizeof(unsigned int) * n);
    double *diff = (double *)malloc(sizeof(double) * n);
    arc_rng rng;
    double base = 0.0;

    arc_rng_seed(&rng, 1);
    arc_sim_robot_info(s, 0, lo, hi, 0);
    for (i = 0; i < n; ++i) {
        ids[i] = i;
        seeds[i] = i;
        diff[i] = 0.7;
    }
    printf("%d worlds, 20 substeps per 1/30 s step\n", n);
    for (t = 0; t < (int)(sizeof(threads) / sizeof(threads[0])); ++t) {
        double t0, el;
        long steps = 0;
        arc_sim_set_threads(s, threads[t]);
        arc_sim_reset(s, ids, n, seeds, diff);
        t0 = omp_get_wtime();
        do {
            for (i = 0; i < n * NJ; ++i)
                targets[i] = arc_rng_range(&rng, lo[i % NJ], hi[i % NJ]);
            arc_sim_step(s, targets);
            steps += n;
            if (steps % (n * 300) == 0)
                arc_sim_reset(s, ids, n, seeds, diff);
            el = omp_get_wtime() - t0;
        } while (el < secs);
        if (t == 0)
            base = steps / el;
        printf("  %2d threads: %9.0f steps/s  (%.2fx, %.1f us/step/thread)\n", threads[t],
               steps / el, steps / el / base, 1e6 * el * threads[t] / steps);
    }
    arc_sim_destroy(s);
    free(targets); free(ids); free(seeds); free(diff);
    return 0;
}
