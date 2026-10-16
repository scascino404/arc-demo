/*
 * arc_phys.h - minimal rigid body physics for articulated robots (C89).
 *
 * The solver is substepped XPBD ("Detailed Rigid Body Simulation with
 * Extended Position Based Dynamics", Mueller et al. 2020):
 *   - rigid bodies with diagonal inertia,
 *   - hinge joints with limits and a torque-limited PD drive,
 *   - collision spheres attached to bodies, colliding only with the terrain,
 *   - terrain made of a grid of axis-aligned columns (a "block" heightfield).
 *
 * A world is a plain struct with fixed capacity and no pointers, so it can
 * be copied with memcpy and many worlds can be stepped in parallel.
 */
#ifndef ARC_PHYS_H
#define ARC_PHYS_H

#include "arc_math.h"

#define ARC_MAX_BODIES 16
#define ARC_MAX_JOINTS 16
#define ARC_MAX_SPHERES 48
#define ARC_MAX_CONTACTS 128
#define ARC_MAX_VISUALS 48
#define ARC_GRID_MAX 64

typedef struct {
    vec3 x, x_prev;    /* center of mass position */
    quat q, q_prev;    /* orientation */
    vec3 v, w;         /* linear / angular velocity (world frame) */
    double mass, inv_mass;
    vec3 inertia, inv_inertia; /* principal moments, body frame */
} arc_body;

typedef struct {
    int body;
    vec3 pos;          /* center, body frame */
    double radius;
    int touch;         /* substeps in contact during the last arc_world_step */
    int anchored;      /* static friction anchor valid (in contact last substep) */
    vec3 anchor;       /* ground point the sphere touched down on */
    vec3 anchor_n;     /* contact normal at the anchor */
} arc_sphere;

typedef struct {
    int parent;        /* -1 attaches the joint to the static world */
    int child;
    vec3 r1, r2;       /* anchor in parent / child frame */
    vec3 a1, a2;       /* hinge axis in parent / child frame */
    vec3 b1, b2;       /* zero-angle reference direction, parent / child frame */
    quat q_rel0;       /* child orientation relative to parent at zero angle */
    double lo, hi;     /* angle limits (lo > hi disables them) */
    double kp, kd;     /* drive stiffness (N m / rad), damping (N m s / rad) */
    double max_torque; /* drive torque limit (N m) */
    double target;     /* drive target angle (rad) */
    /* outputs, refreshed by arc_world_step */
    double angle, rate, torque;
    double torque_acc;
} arc_joint;

typedef struct {
    int body;
    vec3 local;        /* contact point, body frame */
    vec3 n;            /* terrain normal (pointing out of the terrain) */
    double lambda_n;   /* normal impulse of the position solve */
} arc_contact;

typedef struct {
    int nx, ny;
    double cell;       /* column width (m) */
    double ox, oy;     /* world position of the minimum corner of cell (0, 0) */
    double outside;    /* height of the columns outside the grid */
    double h[ARC_GRID_MAX * ARC_GRID_MAX]; /* column heights, h[ix + iy * nx] */
} arc_terrain;

enum { ARC_VIS_SPHERE, ARC_VIS_CAPSULE, ARC_VIS_BOX };

/* Render-only geometry. Ignored by the solver. */
typedef struct {
    int type;
    int body;
    vec3 p0, p1;       /* sphere: p0 center; capsule: p0-p1 segment; box: p0 center, p1 half extents */
    double radius;
    float color[3];
} arc_visual;

typedef struct {
    vec3 gravity;
    double mu_static, mu_dynamic;
    int n_bodies, n_joints, n_spheres, n_contacts, n_visuals;
    int unstable;      /* set when the state blew up (NaN / absurd velocities) */
    double time;
    arc_body bodies[ARC_MAX_BODIES];
    arc_joint joints[ARC_MAX_JOINTS];
    arc_sphere spheres[ARC_MAX_SPHERES];
    arc_contact contacts[ARC_MAX_CONTACTS];
    arc_visual visuals[ARC_MAX_VISUALS];
    arc_terrain terrain;
} arc_world;

void arc_world_init(arc_world *w);

/* Builders. Return the new index, or -1 when the capacity is exhausted. */
int arc_add_body(arc_world *w, double mass, vec3 inertia, vec3 pos, quat q);
int arc_add_sphere(arc_world *w, int body, vec3 local_pos, double radius);
/* Hinge between two bodies as currently posed; that pose is angle zero.
 * parent must be -1 or smaller than child, so that arc_world_set_pose can
 * walk the joints in order. */
int arc_add_hinge(arc_world *w, int parent, int child, vec3 anchor_world, vec3 axis_world);
int arc_add_visual(arc_world *w, int type, int body, vec3 p0, vec3 p1,
                   double radius, float r, float g, float b);

/* Advances by dt using `substeps` XPBD substeps. */
void arc_world_step(arc_world *w, double dt, int substeps);

/* Places body 0 at (pos, q) and every other body by forward kinematics from
 * the given joint angles; zeroes all velocities and sets drive targets to the
 * same angles. Requires every body > 0 to be the child of exactly one joint. */
void arc_world_set_pose(arc_world *w, vec3 pos, quat q, const double *angles);

/* Current joint angle / rate (also cached in the joint after each step). */
double arc_joint_angle(const arc_world *w, const arc_joint *j);
double arc_joint_rate(const arc_world *w, const arc_joint *j);

/* Terrain helpers. */
void arc_terrain_flat(arc_terrain *t, int nx, int ny, double cell, double height);
double arc_terrain_height(const arc_terrain *t, double x, double y);
double arc_terrain_cell(const arc_terrain *t, int ix, int iy);

/* Total mechanical energy (kinetic + gravitational potential). */
double arc_world_energy(const arc_world *w);

/* Inertia helpers (principal moments about the center of mass). */
vec3 arc_inertia_box(double mass, vec3 half_extents);
vec3 arc_inertia_capsule_z(double mass, double radius, double half_length);
vec3 arc_inertia_sphere(double mass, double radius);

#endif
