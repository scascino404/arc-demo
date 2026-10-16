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

/* colors (sRGB) */
static const float ARMOR[3] = { 0.40f, 0.41f, 0.43f };
static const float HULL[3] = { 0.34f, 0.345f, 0.36f };
static const float DARK[3] = { 0.17f, 0.17f, 0.18f };
static const float STEEL[3] = { 0.52f, 0.53f, 0.55f };
static const float BEZEL[3] = { 0.72f, 0.73f, 0.74f };
static const float CABLE[3] = { 0.30f, 0.30f, 0.31f };
static const float BLACK[3] = { 0.06f, 0.06f, 0.065f };
static const float GLOW[3] = { 1.00f, 0.10f, 0.06f };
static const float DIM_GLOW[3] = { 0.62f, 0.05f, 0.03f };

static int visual(arc_world *w, int type, int material, int body, vec3 p0, vec3 p1, double r,
                  double r2, const float *c)
{
    return arc_add_visual(w, type, body, p0, p1, r, r2, material, c[0], c[1], c[2]);
}

/* Armor plate along the body z axis, deeper (body x) than wide (body y). */
static void plate(arc_world *w, int body, vec3 p0, vec3 p1, double r0, double r1, double aspect)
{
    int i = visual(w, ARC_VIS_TAPER, ARC_MAT_PAINT, body, p0, p1, r0, r1, ARMOR);
    if (i >= 0)
        w->visuals[i].aspect = aspect;
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

    /* Visuals, after the ARC Raiders leaper: faceted armor plates over
     * ribbed cable sleeves on the inner side of the leg, bare metal hinges,
     * a small light on each knee and a rubber ball foot. */
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_METAL, bc, v3_scale(d, -0.05), v3_scale(d, 0.035),
           0.062, 0.0, DARK);
    visual(w, ARC_VIS_RING, ARC_MAT_METAL, bc, v3_scale(d, 0.02), d, 0.062, 0.012, STEEL);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_METAL, bf, v3(0, -0.07, -fl), v3(0, 0.07, -fl), 0.045,
           0.0, DARK);
    /* femur */
    plate(w, bf, v3(-0.012, 0, -fl + 0.02), v3(-0.012, 0, fl - 0.02), 0.055, 0.08, 0.8);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_RIBS, bf, v3(0.055, 0, -fl + 0.06), v3(0.07, 0, fl - 0.07),
           0.022, 0.0, CABLE);
    /* knee */
    visual(w, ARC_VIS_BOX, ARC_MAT_PAINT, bf, v3(0, 0, fl), v3(0.085, 0.07, 0.075), 0.0, 0.0, DARK);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_METAL, bf, v3(0, -0.085, fl), v3(0, 0.085, fl), 0.048, 0.0,
           STEEL);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_GLOW, bf, v3(0, 0.085, fl), v3(0, 0.089, fl), 0.02, 0.0,
           GLOW);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_GLOW, bf, v3(0, -0.089, fl), v3(0, -0.085, fl), 0.02, 0.0,
           GLOW);
    /* tibia: two plates meeting in a ridge, tapering down to the ankle */
    plate(w, bt, v3(-0.01, 0, -tl + 0.03), v3(-0.01, 0, -0.05), 0.068, 0.082, 0.8);
    plate(w, bt, v3(-0.01, 0, -0.05), v3(-0.004, 0, tl - 0.11), 0.082, 0.03, 0.85);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_RIBS, bt, v3(0.065, 0, -tl + 0.08), v3(0.045, 0, 0.25 * tl),
           0.02, 0.0, CABLE);
    /* ankle and foot */
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_METAL, bt, v3(0, 0, tl - 0.12), v3(0, 0, tl - 0.035),
           0.026, 0.0, STEEL);
    visual(w, ARC_VIS_RING, ARC_MAT_METAL, bt, v3(0, 0, tl - 0.09), v3(0, 0, 1), 0.03, 0.01, DARK);
    visual(w, ARC_VIS_SPHERE, ARC_MAT_RUBBER, bt, v3(0, 0, tl), v3(0, 0, 0), FOOT_R, 0.0, BLACK);
    visual(w, ARC_VIS_RING, ARC_MAT_METAL, bt, v3(0, 0, tl - 0.02), v3(0, 0, 1), 0.036, 0.011,
           STEEL);
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
    /* paneled hull with collars at the top and bottom, a bottom hatch, two
     * side knobs and the eye: a glossy lens in a silver bezel, ringed by
     * light. The renderer lights the scene from the first glowing ring. */
    visual(w, ARC_VIS_SPHERE, ARC_MAT_PANELS, 0, v3(0, 0, 0), v3(0, 0, 0), BODY_R, 0.0, HULL);
    visual(w, ARC_VIS_RING, ARC_MAT_METAL, 0, v3(0, 0, 0.135), v3(0, 0, 1), 0.148, 0.02, DARK);
    visual(w, ARC_VIS_RING, ARC_MAT_PAINT, 0, v3(0, 0, -0.125), v3(0, 0, 1), 0.156, 0.024, DARK);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_METAL, 0, v3(0, 0, -0.2), v3(0, 0, -0.17), 0.07, 0.0, DARK);
    for (i = -1; i <= 1; i += 2) {
        vec3 k = v3_normalize(v3(0.45, 0.62 * i, -0.64));
        visual(w, ARC_VIS_CYLINDER, ARC_MAT_METAL, 0, v3_scale(k, 0.17), v3_scale(k, 0.215), 0.028,
               0.0, STEEL);
    }
    visual(w, ARC_VIS_RING, ARC_MAT_GLOW, 0, v3(0.18, 0, 0), v3(1, 0, 0), 0.08, 0.013, GLOW);
    visual(w, ARC_VIS_RING, ARC_MAT_METAL, 0, v3(0.168, 0, 0), v3(1, 0, 0), 0.112, 0.02, BEZEL);
    visual(w, ARC_VIS_CYLINDER, ARC_MAT_LENS, 0, v3(0.11, 0, 0), v3(0.178, 0, 0), 0.095, 0.0, BLACK);
    visual(w, ARC_VIS_RING, ARC_MAT_GLOW, 0, v3(0.181, 0, 0), v3(1, 0, 0), 0.032, 0.006, DIM_GLOW);

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
