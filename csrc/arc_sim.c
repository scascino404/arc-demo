/*
 * arc_sim.c - batched simulation of N worlds.
 */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "arc_sim.h"
#include "arc_terrain.h"
#include "arc_rng.h"

struct arc_sim {
    int n;
    int threads;
    double dt;
    int substeps;
    int grid;
    double cell;
    arc_robot_info info;
    arc_world *template_world;  /* robot as built, flat ground */
    arc_world *worlds;
};

arc_sim *arc_sim_create(int n_worlds, double dt, int substeps, int grid, double cell)
{
    arc_sim *s = (arc_sim *)calloc(1, sizeof(arc_sim));
    int i;
    if (!s)
        return 0;
    s->n = n_worlds;
    s->threads = 1;
    s->dt = dt;
    s->substeps = substeps;
    s->grid = grid < ARC_GRID_MAX ? grid : ARC_GRID_MAX;
    s->cell = cell;
    s->template_world = (arc_world *)malloc(sizeof(arc_world));
    s->worlds = (arc_world *)malloc(sizeof(arc_world) * (size_t)n_worlds);
    if (!s->template_world || !s->worlds) {
        arc_sim_destroy(s);
        return 0;
    }
    arc_world_init(s->template_world);
    arc_robot_build(s->template_world, &s->info);
    arc_terrain_flat(&s->template_world->terrain, s->grid, s->grid, cell, 0.0);
    for (i = 0; i < n_worlds; ++i)
        memcpy(&s->worlds[i], s->template_world, sizeof(arc_world));
    return s;
}

void arc_sim_destroy(arc_sim *s)
{
    if (!s)
        return;
    free(s->template_world);
    free(s->worlds);
    free(s);
}

void arc_sim_set_threads(arc_sim *s, int threads)
{
    s->threads = threads > 0 ? threads : 1;
}

int arc_sim_num_worlds(const arc_sim *s) { return s->n; }
int arc_sim_num_joints(void) { return ARC_ROBOT_JOINTS; }
int arc_sim_num_legs(void) { return ARC_ROBOT_LEGS; }

void arc_sim_robot_info(const arc_sim *s, double *default_angles, double *lo, double *hi, double *misc)
{
    int i;
    for (i = 0; i < ARC_ROBOT_JOINTS; ++i) {
        if (default_angles) default_angles[i] = s->info.default_angles[i];
        if (lo) lo[i] = s->info.lo[i];
        if (hi) hi[i] = s->info.hi[i];
    }
    if (misc) {
        misc[0] = s->info.stand_height;
        misc[1] = s->info.total_mass;
    }
}

static void reset_world(const arc_sim *s, arc_world *w, unsigned int seed, double difficulty)
{
    arc_rng rng;
    double angles[ARC_ROBOT_JOINTS], ground = -1e30, yaw, x, y, reach;
    int i;

    arc_rng_seed(&rng, seed);
    /* copy the robot (bodies, joints, spheres, visuals) but not the terrain */
    memcpy(w, s->template_world, offsetof(arc_world, terrain));
    arc_terrain_generate(&w->terrain, s->grid, s->cell, difficulty, &rng);

    for (i = 0; i < ARC_ROBOT_JOINTS; ++i)
        angles[i] = arc_clamp(s->info.default_angles[i] + arc_rng_range(&rng, -0.1, 0.1),
                              s->info.lo[i], s->info.hi[i]);
    yaw = arc_rng_range(&rng, -ARC_PI, ARC_PI);
    /* highest column under the robot's footprint */
    reach = s->info.footprint + 0.1;
    for (x = -reach; x <= reach; x += 0.5 * s->cell)
        for (y = -reach; y <= reach; y += 0.5 * s->cell)
            ground = arc_max(ground, arc_terrain_height(&w->terrain, x, y));
    arc_world_set_pose(w, v3(0.0, 0.0, ground + s->info.stand_height + 0.03),
                       q_axis_angle(v3(0.0, 0.0, 1.0), yaw), angles);
}

void arc_sim_reset(arc_sim *s, const int *ids, int n_ids, const unsigned int *seeds,
                   const double *difficulty)
{
    int k;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(s->threads)
#endif
    for (k = 0; k < n_ids; ++k)
        reset_world(s, &s->worlds[ids[k]], seeds[k], difficulty[k]);
}

void arc_sim_step(arc_sim *s, const double *targets)
{
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(s->threads)
#endif
    for (i = 0; i < s->n; ++i) {
        arc_world *w = &s->worlds[i];
        int j;
        for (j = 0; j < ARC_ROBOT_JOINTS; ++j)
            w->joints[j].target = arc_clamp(targets[i * ARC_ROBOT_JOINTS + j],
                                            w->joints[j].lo, w->joints[j].hi);
        if (!w->unstable)
            arc_world_step(w, s->dt, s->substeps);
    }
}

void arc_sim_observe(const arc_sim *s, double *root, double *rot, double *joints, double *contacts,
                     int *flags)
{
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(s->threads)
#endif
    for (i = 0; i < s->n; ++i) {
        const arc_world *w = &s->worlds[i];
        const arc_body *b = &w->bodies[0];
        int j;
        if (root) {
            double *r = root + i * ARC_SIM_ROOT_SIZE;
            r[0] = b->x.x; r[1] = b->x.y; r[2] = b->x.z;
            r[3] = b->q.w; r[4] = b->q.x; r[5] = b->q.y; r[6] = b->q.z;
            r[7] = b->v.x; r[8] = b->v.y; r[9] = b->v.z;
            r[10] = b->w.x; r[11] = b->w.y; r[12] = b->w.z;
        }
        if (rot) {
            mat3 m;
            m3_from_quat(&m, b->q);
            memcpy(rot + i * ARC_SIM_ROT_SIZE, m.m, sizeof(m.m));
        }
        if (joints) {
            double *o = joints + i * ARC_SIM_JOINT_SIZE;
            for (j = 0; j < ARC_ROBOT_JOINTS; ++j) {
                o[j] = w->joints[j].angle;
                o[ARC_ROBOT_JOINTS + j] = w->joints[j].rate;
                o[2 * ARC_ROBOT_JOINTS + j] = w->joints[j].torque;
            }
        }
        if (contacts) {
            double *o = contacts + i * ARC_SIM_CONTACT_SIZE;
            double torso = 0.0, other = 0.0, inv = 1.0 / s->substeps;
            int is_foot;
            for (j = 0; j < ARC_ROBOT_LEGS; ++j)
                o[j] = w->spheres[s->info.foot_sphere[j]].touch * inv;
            for (j = 0; j < w->n_spheres; ++j) {
                int k;
                double t = w->spheres[j].touch * inv;
                is_foot = 0;
                for (k = 0; k < ARC_ROBOT_LEGS; ++k)
                    is_foot |= s->info.foot_sphere[k] == j;
                if (j < s->info.n_torso_spheres)
                    torso = arc_max(torso, t);
                else if (!is_foot)
                    other = arc_max(other, t);
            }
            o[ARC_ROBOT_LEGS] = torso;
            o[ARC_ROBOT_LEGS + 1] = other;
        }
        if (flags)
            flags[i] = w->unstable;
    }
}

void arc_sim_heights(const arc_sim *s, const double *offsets, int k, double *out)
{
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(s->threads)
#endif
    for (i = 0; i < s->n; ++i) {
        const arc_world *w = &s->worlds[i];
        const arc_body *b = &w->bodies[0];
        vec3 fwd = q_rotate(b->q, v3(1.0, 0.0, 0.0));
        double yaw = atan2(fwd.y, fwd.x), c = cos(yaw), sn = sin(yaw);
        int j;
        for (j = 0; j < k; ++j) {
            double f = offsets[2 * j], l = offsets[2 * j + 1];
            out[i * k + j] = arc_terrain_height(&w->terrain, b->x.x + c * f - sn * l,
                                                b->x.y + sn * f + c * l);
        }
    }
}

void arc_sim_terrain_at(const arc_sim *s, const int *ids, const double *xy, int m, double *out)
{
    int i;
    for (i = 0; i < m; ++i)
        out[i] = arc_terrain_height(&s->worlds[ids[i]].terrain, xy[2 * i], xy[2 * i + 1]);
}

const arc_world *arc_sim_world(const arc_sim *s, int i)
{
    return &s->worlds[i];
}
