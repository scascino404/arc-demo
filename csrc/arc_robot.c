/*
 * arc_robot.c - "leaper"-style spider construction.
 *
 * A spherical body with a glowing eye at the front (+X), hanging below four
 * long legs. Each leg starts on the upper half of the sphere, pointing
 * diagonally outwards; it is built straight and horizontal (all joint angles
 * zero), then the robot is posed in its default stance: femurs raised, knees
 * high above the body, long shins coming down steeply.
 *
 * Femur and tibia bodies are oriented with their local +z along the leg
 * (towards the foot), so their inertia is diagonal; their local +x then
 * faces the inside of the leg in the stance.
 */
#include "arc_robot.h"

#define BODY_MASS 8.0
#define BODY_R 0.2
#define COXA_MASS 0.4
/* Inertia of the coxa including the reflected inertia of the hip motors
 * (rotor inertia times gear ratio squared, "armature" in MuJoCo). Without it
 * the tiny coxa sits between the body and a 1.45 m leg with ~2500x its
 * inertia, which the iterative solver cannot handle. */
#define COXA_INERTIA 0.02
#define FEMUR_MASS 1.5
#define TIBIA_MASS 1.0

#define LEG_ANGLE 0.785    /* legs splay 45 degrees from the body axis */
#define HIP_ELEVATION 0.785 /* hips sit 45 degrees up on the sphere */
#define COXA_LEN 0.06
#define FEMUR_LEN 0.50
#define TIBIA_LEN 0.95
#define FOOT_R 0.045

#define KP 250.0
#define KD 6.0
#define MAX_TORQUE 80.0

static const float HULL[3] = { 0.20f, 0.20f, 0.21f };
static const float DARK[3] = { 0.09f, 0.09f, 0.10f };
static const float STEEL[3] = { 0.42f, 0.42f, 0.44f };
static const float GLOW[3] = { 1.00f, 0.16f, 0.10f };
static const float DIM_GLOW[3] = { 0.62f, 0.07f, 0.05f };

static void visual(arc_world *w, int type, int body, vec3 p0, vec3 p1, double r, double r2,
                   const float *c)
{
    arc_add_visual(w, type, body, p0, p1, r, r2, c == GLOW || c == DIM_GLOW, c[0], c[1], c[2]);
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
    vec3 hip = v3_add(v3_scale(d, BODY_R * cos(HIP_ELEVATION)), v3(0, 0, BODY_R * sin(HIP_ELEVATION)));
    vec3 p1 = v3_madd(hip, d, COXA_LEN);           /* hip pitch axis */
    vec3 p2 = v3_madd(p1, d, FEMUR_LEN);           /* knee axis */
    double fl = 0.5 * FEMUR_LEN, tl = 0.5 * TIBIA_LEN;
    int bc, bf, bt, j;

    bc = arc_add_body(w, COXA_MASS, v3(COXA_INERTIA, COXA_INERTIA, COXA_INERTIA),
                      v3_madd(hip, d, 0.5 * COXA_LEN), q_identity());
    bf = arc_add_body(w, FEMUR_MASS, arc_inertia_capsule_z(FEMUR_MASS, 0.06, fl),
                      v3_madd(p1, d, fl), along);
    bt = arc_add_body(w, TIBIA_MASS, arc_inertia_capsule_z(TIBIA_MASS, 0.05, tl),
                      v3_madd(p2, d, tl), along);

    /* yaw about -z on the left, +z on the right: positive swings forwards */
    j = arc_add_hinge(w, 0, bc, hip, v3(0.0, 0.0, -sy));
    set_drive(&w->joints[j], -0.8, 0.8);
    j = arc_add_hinge(w, bc, bf, p1, lift);
    set_drive(&w->joints[j], -0.4, 1.5);
    j = arc_add_hinge(w, bf, bt, p2, lift);
    set_drive(&w->joints[j], -2.8, -0.5);

    info->default_angles[3 * leg + 0] = 0.0;
    info->default_angles[3 * leg + 1] = 0.6;
    info->default_angles[3 * leg + 2] = -1.95;

    /* collision: femur middle, knee, two along the tibia, foot */
    arc_add_sphere(w, bf, v3(0.0, 0.0, 0.0), 0.06);
    arc_add_sphere(w, bt, v3(0.0, 0.0, -tl), 0.07);
    arc_add_sphere(w, bt, v3(0.0, 0.0, -0.3 * tl), 0.055);
    arc_add_sphere(w, bt, v3(0.0, 0.0, 0.4 * tl), 0.035);
    info->foot_sphere[leg] = arc_add_sphere(w, bt, v3(0.0, 0.0, tl), FOOT_R);

    /* hip knob */
    visual(w, ARC_VIS_CYLINDER, bc, v3_scale(d, -0.5 * COXA_LEN), v3_scale(d, 0.5 * COXA_LEN),
           0.045, 0.0, STEEL);
    /* femur: angular, thickening towards a boxy knee */
    visual(w, ARC_VIS_TAPER, bf, v3(0, 0, -fl), v3(0, 0, fl - 0.06), 0.045, 0.07, HULL);
    visual(w, ARC_VIS_BOX, bf, v3(0, 0, fl), v3(0.08, 0.075, 0.08), 0.0, 0.0, DARK);
    visual(w, ARC_VIS_BOX, bf, v3(0.081, 0, fl), v3(0.004, 0.035, 0.05), 0.0, 0.0, GLOW);
    /* tibia: a blade tapering down to the foot, glowing on its inner face */
    visual(w, ARC_VIS_TAPER, bt, v3(0, 0, -tl + 0.04), v3(0, 0, tl - 0.06), 0.075, 0.022, HULL);
    visual(w, ARC_VIS_TAPER, bt, v3(0.05, 0, -tl + 0.12), v3(0.016, 0, 0.1 * tl), 0.018, 0.006,
           DIM_GLOW);
    visual(w, ARC_VIS_CYLINDER, bt, v3(0, 0, tl - 0.075), v3(0, 0, tl - 0.04), 0.035, 0.0, STEEL);
    visual(w, ARC_VIS_SPHERE, bt, v3(0, 0, tl), v3(0, 0, 0), FOOT_R, 0.0, DARK);
}

void arc_robot_build(arc_world *w, arc_robot_info *info)
{
    int leg, i;
    double lowest = 1e30;
    static const double SX[ARC_ROBOT_LEGS] = { 1.0, 1.0, -1.0, -1.0 };
    static const double SY[ARC_ROBOT_LEGS] = { 1.0, -1.0, 1.0, -1.0 };

    arc_add_body(w, BODY_MASS, arc_inertia_sphere(BODY_MASS, BODY_R), v3(0, 0, 0), q_identity());
    arc_add_sphere(w, 0, v3(0, 0, 0), BODY_R);
    info->n_torso_spheres = w->n_spheres;
    /* hull, eye (dark lens in a glowing ring) and a band around the bottom */
    visual(w, ARC_VIS_SPHERE, 0, v3(0, 0, 0), v3(0, 0, 0), BODY_R, 0.0, HULL);
    visual(w, ARC_VIS_CYLINDER, 0, v3(0.12, 0, 0), v3(0.2, 0, 0), 0.13, 0.0, DARK);
    visual(w, ARC_VIS_RING, 0, v3(0.204, 0, 0), v3(1, 0, 0), 0.105, 0.021, GLOW);
    visual(w, ARC_VIS_RING, 0, v3(0.201, 0, 0), v3(1, 0, 0), 0.05, 0.008, DIM_GLOW);
    visual(w, ARC_VIS_RING, 0, v3(0, 0, -0.12), v3(0, 0, 1), 0.158, 0.014, STEEL);

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
