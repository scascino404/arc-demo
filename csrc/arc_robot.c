/*
 * arc_robot.c - spider-like quadruped construction.
 *
 * Each leg is built straight and horizontal, pointing diagonally outwards
 * from a torso corner (all joint angles zero), then the robot is posed in
 * its default stance: femurs raised, knees bent sharply down.
 *
 * Femur and tibia bodies are oriented with their local +z along the leg
 * (towards the foot), so their capsule inertia is diagonal.
 */
#include "arc_robot.h"

#define TORSO_MASS 10.0
#define COXA_MASS 0.6
#define FEMUR_MASS 1.5
#define TIBIA_MASS 0.8

#define HIP_X 0.26         /* hip yaw axes in the torso frame */
#define HIP_Y 0.16
#define LEG_ANGLE 0.785    /* legs splay 45 degrees from the body axis */
#define COXA_LEN 0.10
#define FEMUR_LEN 0.45
#define TIBIA_LEN 0.62
#define FOOT_R 0.04

#define KP 200.0
#define KD 5.0
#define MAX_TORQUE 80.0

static const float BODY_COLOR[3] = { 0.30f, 0.76f, 0.86f };
static const float JOINT_COLOR[3] = { 0.20f, 0.52f, 0.62f };
static const float DARK_COLOR[3] = { 0.12f, 0.14f, 0.16f };

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
    double psi = sx > 0.0 ? sy * LEG_ANGLE : sy * (ARC_PI - LEG_ANGLE);
    vec3 d = v3(cos(psi), sin(psi), 0.0);          /* leg direction */
    vec3 up = v3(0.0, 0.0, 1.0);
    vec3 lift = v3_cross(d, up);                   /* positive rotation raises the leg */
    quat along = q_axis_angle(v3_normalize(v3_cross(up, d)), 0.5 * ARC_PI); /* z -> d */
    vec3 hip = v3(sx * HIP_X, sy * HIP_Y, 0.0);
    vec3 p1 = v3_madd(hip, d, COXA_LEN);           /* hip pitch axis */
    vec3 p2 = v3_madd(p1, d, FEMUR_LEN);           /* knee axis */
    double fl = 0.5 * FEMUR_LEN, tl = 0.5 * TIBIA_LEN;
    int bc, bf, bt, j;

    bc = arc_add_body(w, COXA_MASS, arc_inertia_sphere(COXA_MASS, 0.06),
                      v3_madd(hip, d, 0.5 * COXA_LEN), q_identity());
    bf = arc_add_body(w, FEMUR_MASS, arc_inertia_capsule_z(FEMUR_MASS, 0.05, fl),
                      v3_madd(p1, d, fl), along);
    bt = arc_add_body(w, TIBIA_MASS, arc_inertia_capsule_z(TIBIA_MASS, 0.035, tl),
                      v3_madd(p2, d, tl), along);

    /* yaw about -z on the left, +z on the right: positive swings forwards */
    j = arc_add_hinge(w, 0, bc, hip, v3(0.0, 0.0, -sy));
    set_drive(&w->joints[j], -0.8, 0.8);
    j = arc_add_hinge(w, bc, bf, p1, lift);
    set_drive(&w->joints[j], -0.5, 1.4);
    j = arc_add_hinge(w, bf, bt, p2, lift);
    set_drive(&w->joints[j], -2.7, -0.3);

    info->default_angles[3 * leg + 0] = 0.0;
    info->default_angles[3 * leg + 1] = 0.6;
    info->default_angles[3 * leg + 2] = -1.95;

    /* collision: femur middle, knee, tibia middle, foot */
    arc_add_sphere(w, bf, v3(0.0, 0.0, 0.0), 0.05);
    arc_add_sphere(w, bt, v3(0.0, 0.0, -tl), 0.055);
    arc_add_sphere(w, bt, v3(0.0, 0.0, 0.0), 0.035);
    info->foot_sphere[leg] = arc_add_sphere(w, bt, v3(0.0, 0.0, tl), FOOT_R);

    visual(w, ARC_VIS_CAPSULE, bc, v3_scale(d, -0.5 * COXA_LEN), v3_scale(d, 0.5 * COXA_LEN),
           0.065, JOINT_COLOR);
    visual(w, ARC_VIS_SPHERE, bf, v3(0.0, 0.0, -fl), v3(0, 0, 0), 0.07, JOINT_COLOR);
    visual(w, ARC_VIS_CAPSULE, bf, v3(0.0, 0.0, -fl), v3(0.0, 0.0, fl), 0.05, BODY_COLOR);
    visual(w, ARC_VIS_SPHERE, bt, v3(0.0, 0.0, -tl), v3(0, 0, 0), 0.06, JOINT_COLOR);
    visual(w, ARC_VIS_CAPSULE, bt, v3(0.0, 0.0, -tl), v3(0.0, 0.0, 0.3 * tl), 0.04, BODY_COLOR);
    visual(w, ARC_VIS_CAPSULE, bt, v3(0.0, 0.0, 0.3 * tl), v3(0.0, 0.0, tl), 0.025, BODY_COLOR);
    visual(w, ARC_VIS_SPHERE, bt, v3(0.0, 0.0, tl), v3(0, 0, 0), FOOT_R, DARK_COLOR);
}

void arc_robot_build(arc_world *w, arc_robot_info *info)
{
    vec3 he = v3(0.30, 0.15, 0.08);
    int leg, i, ix, iy;
    double lowest = 1e30;
    static const double SX[ARC_ROBOT_LEGS] = { 1.0, 1.0, -1.0, -1.0 };
    static const double SY[ARC_ROBOT_LEGS] = { 1.0, -1.0, 1.0, -1.0 };

    arc_add_body(w, TORSO_MASS, arc_inertia_box(TORSO_MASS, he), v3(0, 0, 0), q_identity());
    for (ix = -1; ix <= 1; ++ix)
        for (iy = -1; iy <= 1; iy += 2)
            arc_add_sphere(w, 0, v3(ix * 0.2, iy * 0.12, 0.0), 0.1);
    info->n_torso_spheres = w->n_spheres;
    /* core block, two side pods and a sensor head */
    visual(w, ARC_VIS_BOX, 0, v3(0, 0, 0), he, 0.0, JOINT_COLOR);
    visual(w, ARC_VIS_CAPSULE, 0, v3(-0.2, 0.15, 0.02), v3(0.2, 0.15, 0.02), 0.1, BODY_COLOR);
    visual(w, ARC_VIS_CAPSULE, 0, v3(-0.2, -0.15, 0.02), v3(0.2, -0.15, 0.02), 0.1, BODY_COLOR);
    visual(w, ARC_VIS_BOX, 0, v3(0.31, 0.0, 0.03), v3(0.05, 0.07, 0.05), 0.0, DARK_COLOR);

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
    info->footprint = 0.0;
    for (leg = 0; leg < ARC_ROBOT_LEGS; ++leg) {
        const arc_sphere *s = &w->spheres[info->foot_sphere[leg]];
        const arc_body *b = &w->bodies[s->body];
        vec3 c = v3_add(b->x, q_rotate(b->q, s->pos));
        lowest = arc_min(lowest, c.z - s->radius);
        info->footprint += sqrt(c.x * c.x + c.y * c.y) / ARC_ROBOT_LEGS;
    }
    info->stand_height = -lowest;
    arc_world_set_pose(w, v3(0, 0, info->stand_height), q_identity(), info->default_angles);
}
