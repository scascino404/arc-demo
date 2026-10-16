/*
 * arc_robot.c - quadruped construction.
 *
 * The robot is built with straight legs pointing down (all joint angles
 * zero) and then posed in its default stance: front knees bend backwards,
 * hind knees forwards.
 */
#include "arc_robot.h"

#define TORSO_MASS 10.0
#define HIP_MASS 1.0
#define THIGH_MASS 1.6
#define SHIN_MASS 0.5

#define HIP_X 0.30         /* hip positions in the torso frame */
#define HIP_Y 0.15
#define HIP_LEN 0.10       /* abduction axis to flexion axis, sideways */
#define THIGH_LEN 0.34
#define SHIN_LEN 0.34
#define FOOT_R 0.04

#define KP 150.0
#define KD 4.0
#define MAX_TORQUE 60.0

static const float BODY_COLOR[3] = { 0.25f, 0.72f, 0.82f };
static const float JOINT_COLOR[3] = { 0.18f, 0.48f, 0.58f };
static const float FOOT_COLOR[3] = { 0.12f, 0.14f, 0.16f };

static void visual(arc_world *w, int type, int body, vec3 p0, vec3 p1, double r, const float *c)
{
    arc_add_visual(w, type, body, p0, p1, r, c[0], c[1], c[2]);
}

static void set_drive(arc_joint *j, double lo, double hi)
{
    j->lo = lo;
    j->hi = hi;
    j->kp = KP;
    j->kd = KD;
    j->max_torque = MAX_TORQUE;
}

static void build_leg(arc_world *w, arc_robot_info *info, int leg, double sx, double sy)
{
    vec3 hip = v3(sx * HIP_X, sy * HIP_Y, 0.0);
    vec3 top = v3_add(hip, v3(0.0, sy * HIP_LEN, 0.0));
    vec3 knee = v3_add(top, v3(0.0, 0.0, -THIGH_LEN));
    double hl = 0.5 * THIGH_LEN, sl = 0.5 * SHIN_LEN;
    int bh, bt, bs, j;
    int front = sx > 0.0;

    bh = arc_add_body(w, HIP_MASS, arc_inertia_sphere(HIP_MASS, 0.06),
                      v3_add(hip, v3(0.0, sy * 0.5 * HIP_LEN, 0.0)), q_identity());
    bt = arc_add_body(w, THIGH_MASS, arc_inertia_capsule_z(THIGH_MASS, 0.045, hl),
                      v3_add(top, v3(0.0, 0.0, -hl)), q_identity());
    bs = arc_add_body(w, SHIN_MASS, arc_inertia_capsule_z(SHIN_MASS, 0.03, sl),
                      v3_add(knee, v3(0.0, 0.0, -sl)), q_identity());

    /* abduction: positive moves the foot outwards on both sides */
    j = arc_add_hinge(w, 0, bh, hip, v3(sy, 0.0, 0.0));
    set_drive(&w->joints[j], -0.5, 0.7);
    /* flexion and knee: positive swings the distal segment backwards */
    j = arc_add_hinge(w, bh, bt, top, v3(0.0, 1.0, 0.0));
    set_drive(&w->joints[j], -1.6, 1.6);
    j = arc_add_hinge(w, bt, bs, knee, v3(0.0, 1.0, 0.0));
    if (front)
        set_drive(&w->joints[j], -2.5, 0.0);
    else
        set_drive(&w->joints[j], 0.0, 2.5);

    info->default_angles[3 * leg + 0] = 0.0;
    info->default_angles[3 * leg + 1] = front ? 0.6 : -0.6;
    info->default_angles[3 * leg + 2] = front ? -1.2 : 1.2;

    /* collision: thigh middle, knee, shin middle, foot */
    arc_add_sphere(w, bt, v3(0.0, 0.0, 0.0), 0.045);
    arc_add_sphere(w, bs, v3(0.0, 0.0, sl), 0.045);
    arc_add_sphere(w, bs, v3(0.0, 0.0, 0.0), 0.03);
    info->foot_sphere[leg] = arc_add_sphere(w, bs, v3(0.0, 0.0, -sl), FOOT_R);

    visual(w, ARC_VIS_CAPSULE, bh, v3(0.0, -sy * 0.5 * HIP_LEN, 0.0),
           v3(0.0, sy * 0.5 * HIP_LEN, 0.0), 0.06, JOINT_COLOR);
    visual(w, ARC_VIS_CAPSULE, bt, v3(0.0, 0.0, hl), v3(0.0, 0.0, -hl), 0.045, BODY_COLOR);
    visual(w, ARC_VIS_SPHERE, bs, v3(0.0, 0.0, sl), v3(0, 0, 0), 0.05, JOINT_COLOR);
    visual(w, ARC_VIS_CAPSULE, bs, v3(0.0, 0.0, sl), v3(0.0, 0.0, -sl), 0.03, BODY_COLOR);
    visual(w, ARC_VIS_SPHERE, bs, v3(0.0, 0.0, -sl), v3(0, 0, 0), FOOT_R, FOOT_COLOR);
}

void arc_robot_build(arc_world *w, arc_robot_info *info)
{
    vec3 he = v3(0.35, 0.15, 0.08);
    int leg, i, ix, iy;
    double lowest = 1e30;
    static const double SX[ARC_ROBOT_LEGS] = { 1.0, 1.0, -1.0, -1.0 };
    static const double SY[ARC_ROBOT_LEGS] = { 1.0, -1.0, 1.0, -1.0 };

    arc_add_body(w, TORSO_MASS, arc_inertia_box(TORSO_MASS, he), v3(0, 0, 0), q_identity());
    for (ix = -1; ix <= 1; ++ix)
        for (iy = -1; iy <= 1; iy += 2)
            arc_add_sphere(w, 0, v3(ix * (he.x - he.z), iy * (he.y - he.z), 0.0), he.z);
    info->n_torso_spheres = w->n_spheres;
    visual(w, ARC_VIS_BOX, 0, v3(0, 0, 0), he, 0.0, BODY_COLOR);
    visual(w, ARC_VIS_BOX, 0, v3(he.x, 0, 0.02), v3(0.03, 0.08, 0.04), 0.0, FOOT_COLOR);

    for (leg = 0; leg < ARC_ROBOT_LEGS; ++leg)
        build_leg(w, info, leg, SX[leg], SY[leg]);

    for (i = 0; i < ARC_ROBOT_JOINTS; ++i) {
        info->lo[i] = w->joints[i].lo;
        info->hi[i] = w->joints[i].hi;
    }
    info->total_mass = 0.0;
    for (i = 0; i < w->n_bodies; ++i)
        info->total_mass += w->bodies[i].mass;

    /* stand height: put the feet on z = 0 in the default stance */
    arc_world_set_pose(w, v3(0, 0, 0), q_identity(), info->default_angles);
    for (leg = 0; leg < ARC_ROBOT_LEGS; ++leg) {
        const arc_sphere *s = &w->spheres[info->foot_sphere[leg]];
        const arc_body *b = &w->bodies[s->body];
        lowest = arc_min(lowest, v3_add(b->x, q_rotate(b->q, s->pos)).z - s->radius);
    }
    info->stand_height = -lowest;
    arc_world_set_pose(w, v3(0, 0, info->stand_height), q_identity(), info->default_angles);
}
