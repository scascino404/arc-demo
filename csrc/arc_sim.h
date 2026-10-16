/*
 * arc_sim.h - batched simulation API, the boundary with Python (ctypes).
 *
 * A simulation holds N independent worlds, each with the quadruped and its
 * own terrain. Every call works on all worlds at once with flat arrays laid
 * out world-major, and stepping runs the worlds in parallel with OpenMP.
 * The task logic (targets, rewards, terminations) lives in Python.
 */
#ifndef ARC_SIM_H
#define ARC_SIM_H

#include "arc_phys.h"
#include "arc_robot.h"

typedef struct arc_sim arc_sim;

/* Sizes of the per-world blocks returned by arc_sim_observe. */
#define ARC_SIM_ROOT_SIZE 13               /* pos(3) quat wxyz(4) linvel(3) angvel(3), world frame */
#define ARC_SIM_JOINT_SIZE (3 * ARC_ROBOT_JOINTS) /* angles, rates, torques */
#define ARC_SIM_CONTACT_SIZE (ARC_ROBOT_LEGS + 2) /* feet, torso, other leg parts */

/* dt: control period; substeps: XPBD substeps per control period;
 * grid x cell: terrain size. */
arc_sim *arc_sim_create(int n_worlds, double dt, int substeps, int grid, double cell);
void arc_sim_destroy(arc_sim *s);
void arc_sim_set_threads(arc_sim *s, int threads);
int arc_sim_num_worlds(const arc_sim *s);
int arc_sim_num_joints(void);
int arc_sim_num_legs(void);

/* default_angles, lo, hi: ARC_ROBOT_JOINTS each; misc: stand height, mass. */
void arc_sim_robot_info(const arc_sim *s, double *default_angles, double *lo, double *hi, double *misc);

/* Regenerates the terrain of the listed worlds (seed, difficulty in [0, 1])
 * and drops the robot at the origin in its default stance with a random
 * yaw and a little joint noise. */
void arc_sim_reset(arc_sim *s, const int *ids, int n_ids, const unsigned int *seeds,
                   const double *difficulty);

/* targets: N x ARC_ROBOT_JOINTS drive targets (rad), clamped to the limits. */
void arc_sim_step(arc_sim *s, const double *targets);

/* Any output pointer may be NULL. contacts are the fraction of substeps in
 * contact during the last step; flags is 1 for worlds that became unstable. */
void arc_sim_observe(const arc_sim *s, double *root, double *joints, double *contacts, int *flags);

/* Terrain heights at k points given as (forward, left) offsets from the
 * torso in its heading frame. out: N x k. */
void arc_sim_heights(const arc_sim *s, const double *offsets, int k, double *out);

/* Terrain height at m (x, y) world points, point i in world ids[i]. */
void arc_sim_terrain_at(const arc_sim *s, const int *ids, const double *xy, int m, double *out);

const arc_world *arc_sim_world(const arc_sim *s, int i);

#endif
