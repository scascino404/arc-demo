/*
 * test_phys.c - physics regression tests: analytic cases first (free fall,
 * free rotation, pendulum), then constraint quality (chains, limits, drives)
 * and contacts (resting, sliding friction, block terrain).
 */
#include "../arc_phys.h"
#include "check.h"

static void run(arc_world *w, double seconds)
{
    int n = (int)(seconds / DT + 0.5), i;
    for (i = 0; i < n; ++i)
        arc_world_step(w, DT, SUBSTEPS);
}

/* World with the ground far away, so nothing collides. */
static void init_empty(arc_world *w)
{
    arc_world_init(w);
    arc_terrain_flat(&w->terrain, 1, 1, 1.0, -1e6);
}

static void test_free_fall(void)
{
    arc_world w;
    printf("free fall\n");
    init_empty(&w);
    arc_add_body(&w, 2.0, arc_inertia_sphere(2.0, 0.1), v3(0, 0, 0), q_identity());
    w.bodies[0].v = v3(1.0, 0.0, 0.0);
    run(&w, 1.0);
    /* symplectic Euler: error g h t / 2 = 8 mm with h = 1/600 */
    check_near("z after 1s", w.bodies[0].x.z, -0.5 * 9.81, 1e-2);
    check_near("x after 1s", w.bodies[0].x.x, 1.0, 1e-9);
    check_near("vz after 1s", w.bodies[0].v.z, -9.81, 1e-9);
}

static void test_free_rotation(void)
{
    arc_world w;
    arc_body *b;
    vec3 L0, L1, i0;
    double e0, e1;
    printf("free rotation (gyroscopic)\n");
    init_empty(&w);
    w.gravity = v3(0, 0, 0);
    arc_add_body(&w, 1.0, v3(1.0, 2.0, 3.0), v3(0, 0, 0), q_identity());
    b = &w.bodies[0];
    b->w = v3(0.2, 0.3, 4.0);
    i0 = v3_mul(q_inv_rotate(b->q, b->w), b->inertia);
    L0 = q_rotate(b->q, i0);
    e0 = arc_world_energy(&w);
    run(&w, 5.0);
    L1 = q_rotate(b->q, v3_mul(q_inv_rotate(b->q, b->w), b->inertia));
    e1 = arc_world_energy(&w);
    check_below("angular momentum drift (rel)", v3_len(v3_sub(L1, L0)) / v3_len(L0), 0.02);
    /* the implicit gyroscopic step is slightly dissipative, never energetic */
    check_below("energy gain (rel)", (e1 - e0) / e0, 1e-6);
    check_below("energy loss (rel)", (e0 - e1) / e0, 0.05);
}

static void test_pendulum(void)
{
    arc_world w;
    double len = 1.0, m = 1.0, r = 0.05, theta0 = 0.1;
    double i_pivot, period, a_prev, a, e0, e1;
    int i, crossings = 0, steps;
    double first = -1.0, last = 0.0;
    printf("pendulum\n");
    init_empty(&w);
    arc_add_body(&w, m, arc_inertia_sphere(m, r), v3(0, 0, -len), q_identity());
    arc_add_hinge(&w, -1, 0, v3(0, 0, 0), v3(0, 1, 0));
    /* release from theta0 */
    arc_world_set_pose(&w, v3(0, 0, 0), q_identity(), &theta0);
    check_near("FK pendulum x", w.bodies[0].x.x, -len * sin(theta0), 1e-12);
    i_pivot = 0.4 * m * r * r + m * len * len;
    period = 2.0 * ARC_PI * sqrt(i_pivot / (m * 9.81 * len));
    e0 = arc_world_energy(&w);
    a_prev = arc_joint_angle(&w, &w.joints[0]);
    check_near("initial angle", a_prev, theta0, 1e-9);
    /* count downward zero crossings with a fine step */
    steps = (int)(10.0 * 600.0);
    for (i = 0; i < steps; ++i) {
        arc_world_step(&w, 1.0 / 600.0, 1);
        a = arc_joint_angle(&w, &w.joints[0]);
        if (a_prev > 0.0 && a <= 0.0) {
            double t = w.time - (1.0 / 600.0) * a / (a - a_prev);
            if (first < 0.0)
                first = t;
            last = t;
            ++crossings;
        }
        a_prev = a;
    }
    check_near("period (s)", (last - first) / (crossings - 1), period, 2e-3 * period);
    e1 = arc_world_energy(&w);
    /* first order XPBD is dissipative: allow a slow loss, forbid any gain */
    check_below("energy gain after 10 s (rel to swing)",
                (e1 - e0) / (m * 9.81 * len * (1.0 - cos(theta0))), 1e-6);
    check_below("energy loss after 10 s (rel to swing)",
                (e0 - e1) / (m * 9.81 * len * (1.0 - cos(theta0))), 0.2);
}

static void test_chain(void)
{
    arc_world w;
    int i, n = 6;
    double seg = 0.3, maxsep = 0.0;
    printf("hanging chain of %d links\n", n);
    init_empty(&w);
    for (i = 0; i < n; ++i) {
        /* horizontal chain along +x, link mass decreasing 10:1 */
        double m = 1.0 - 0.9 * i / (n - 1.0);
        arc_add_body(&w, m, arc_inertia_capsule_z(m, 0.03, 0.5 * seg),
                     v3((i + 0.5) * seg, 0, 0), q_axis_angle(v3(0, 1, 0), 0.5 * ARC_PI));
        arc_add_hinge(&w, i - 1, i, v3(i * seg, 0, 0), v3(0, 1, 0));
    }
    for (i = 0; i < 90; ++i) {
        int k;
        arc_world_step(&w, DT, SUBSTEPS);
        for (k = 0; k < w.n_joints; ++k) {
            arc_joint *j = &w.joints[k];
            vec3 p1 = j->parent >= 0 ? v3_add(w.bodies[j->parent].x, q_rotate(w.bodies[j->parent].q, j->r1)) : j->r1;
            vec3 p2 = v3_add(w.bodies[j->child].x, q_rotate(w.bodies[j->child].q, j->r2));
            maxsep = arc_max(maxsep, v3_len(v3_sub(p1, p2)));
        }
    }
    check_below("max joint separation (m)", maxsep, 2e-3);
    check(!w.unstable, "stable", w.unstable, 0);
    /* after swinging for 3 s with damping-free joints energy must not grow */
    check_below("tip height <= release height", w.bodies[n - 1].x.z, 1e-6);
}

static void test_limits(void)
{
    arc_world w;
    double lo = -0.3, hi = 0.4, amin = 1e9, amax = -1e9;
    int i;
    printf("hinge limits\n");
    init_empty(&w);
    /* horizontal arm along +x: gravity swings it to positive angles about +y */
    arc_add_body(&w, 1.0, arc_inertia_capsule_z(1.0, 0.03, 0.25), v3(0.25, 0, 0),
                 q_axis_angle(v3(0, 1, 0), 0.5 * ARC_PI));
    arc_add_hinge(&w, -1, 0, v3(0, 0, 0), v3(0, 1, 0));
    w.joints[0].lo = lo;
    w.joints[0].hi = hi;
    for (i = 0; i < 120; ++i) {
        arc_world_step(&w, DT, SUBSTEPS);
        amin = arc_min(amin, w.joints[0].angle);
        amax = arc_max(amax, w.joints[0].angle);
    }
    check_below("max angle overshoot (rad)", amax - hi, 5e-3);
    check_below("min angle undershoot (rad)", lo - amin, 5e-3);
    check_near("rests on upper limit", w.joints[0].angle, hi, 5e-3);
}

static void test_drive(void)
{
    arc_world w;
    double m = 1.0, len = 0.5, target = 0.5;
    printf("PD drive\n");
    init_empty(&w);
    w.gravity = v3(0, 0, 0);
    arc_add_body(&w, m, arc_inertia_capsule_z(m, 0.03, 0.5 * len), v3(0.5 * len, 0, 0),
                 q_axis_angle(v3(0, 1, 0), 0.5 * ARC_PI));
    arc_add_hinge(&w, -1, 0, v3(0, 0, 0), v3(0, 1, 0));
    w.joints[0].kp = 50.0;
    w.joints[0].kd = 2.0;
    w.joints[0].max_torque = 100.0;
    w.joints[0].target = target;
    run(&w, 2.0);
    check_near("reaches target (no gravity)", w.joints[0].angle, target, 1e-3);
    check_near("settled rate", w.joints[0].rate, 0.0, 1e-3);

    /* with gravity the static sag is m g (L/2) / kp; arm held horizontal */
    w.gravity = v3(0, 0, -9.81);
    w.joints[0].target = 0.0;
    run(&w, 3.0);
    check_near("gravity sag", w.joints[0].angle, m * 9.81 * 0.5 * len / 50.0, 2e-3);
    check_near("holding torque", w.joints[0].torque, -m * 9.81 * 0.5 * len, 0.05);

    /* a weak motor saturates and the arm falls to hang down */
    w.joints[0].max_torque = 0.2;
    run(&w, 6.0);
    check_near("saturated torque", fabs(w.joints[0].torque), 0.2, 0.02);
    check(w.joints[0].angle > 1.2, "weak motor cannot hold the arm", w.joints[0].angle, 1.2);
}

static void test_resting_sphere(void)
{
    arc_world w;
    double r = 0.1, maxv = 0.0;
    int i;
    printf("resting sphere\n");
    arc_world_init(&w);
    arc_terrain_flat(&w.terrain, 8, 8, 0.5, 0.0);
    arc_add_body(&w, 1.0, arc_inertia_sphere(1.0, r), v3(0.01, 0.02, 0.5), q_identity());
    arc_add_sphere(&w, 0, v3(0, 0, 0), r);
    run(&w, 1.0);
    for (i = 0; i < 60; ++i) {
        arc_world_step(&w, DT, SUBSTEPS);
        maxv = arc_max(maxv, v3_len(w.bodies[0].v));
    }
    check_near("rest height", w.bodies[0].x.z, r, 1e-4);
    check_below("max speed while resting", maxv, 1e-3);
    check(w.spheres[0].touch == SUBSTEPS, "touching every substep", w.spheres[0].touch, SUBSTEPS);
}

static void add_box_body(arc_world *w, double m, vec3 he, vec3 pos, double r)
{
    int b = arc_add_body(w, m, arc_inertia_box(m, he), pos, q_identity());
    int sx, sy;
    for (sx = -1; sx <= 1; sx += 2)
        for (sy = -1; sy <= 1; sy += 2)
            arc_add_sphere(w, b, v3(sx * (he.x - r), sy * (he.y - r), -he.z + r), r);
}

static void test_sliding(void)
{
    arc_world w;
    double v0 = 3.0, mu = 0.5, x0;
    printf("sliding friction\n");
    arc_world_init(&w);
    w.mu_dynamic = mu;
    w.mu_static = mu;
    arc_terrain_flat(&w.terrain, 32, 32, 0.5, 0.0);
    add_box_body(&w, 2.0, v3(0.2, 0.2, 0.05), v3(-4.0, 0, 0.05), 0.03);
    run(&w, 0.2);
    x0 = w.bodies[0].x.x;
    w.bodies[0].v = v3(v0, 0, 0);
    run(&w, 2.0);
    check_near("stopping distance (m)", w.bodies[0].x.x - x0, v0 * v0 / (2.0 * mu * 9.81), 0.05);
    check_near("stopped", v3_len(w.bodies[0].v), 0.0, 5e-3);

    /* on a 20 degree slope, approximated by tilted gravity: mu_s = 0.5
     * > tan(20 deg) = 0.36, so the box must stay put */
    {
        double x1 = w.bodies[0].x.x, ang = 20.0 * ARC_PI / 180.0;
        w.gravity = v3(9.81 * sin(ang), 0, -9.81 * cos(ang));
        run(&w, 2.0);
        /* Gauss-Seidel friction creeps slightly near the cone limit (here at
         * 73% of it): accept up to 2.5 mm/s */
        check_below("static friction holds (drift m)", fabs(w.bodies[0].x.x - x1), 5e-3);
        /* mu_s = 0.3 < tan(20 deg): it must slide */
        w.mu_static = w.mu_dynamic = 0.3;
        run(&w, 1.0);
        /* a = g (sin - mu cos) = 0.59 m/s^2: 0.29 m in the first second */
        check(w.bodies[0].x.x - x1 > 0.25, "slides when mu_s < tan(slope)", w.bodies[0].x.x - x1, 0.25);
    }
}

static void test_blocks(void)
{
    arc_world w;
    double r = 0.1;
    arc_terrain *t = &w.terrain;
    int i;
    printf("block terrain\n");
    arc_world_init(&w);
    arc_terrain_flat(t, 8, 8, 0.5, 0.0);
    /* a 0.3 m high plateau over the cells with ix >= 4 (x >= 0) */
    for (i = 0; i < 64; ++i)
        if (i % 8 >= 4)
            t->h[i] = 0.3;
    check_near("height lookup plateau", arc_terrain_height(t, 0.1, 0.1), 0.3, 1e-12);
    check_near("height lookup ground", arc_terrain_height(t, -0.1, 0.1), 0.0, 1e-12);

    /* sphere resting across the seam between two plateau cells */
    arc_add_body(&w, 1.0, arc_inertia_sphere(1.0, r), v3(0.5, 0.0, 0.6), q_identity());
    arc_add_sphere(&w, 0, v3(0, 0, 0), r);
    /* sphere dropped right next to the step: must fall to the ground */
    arc_add_body(&w, 1.0, arc_inertia_sphere(1.0, r), v3(-0.12, 0.25, 0.6), q_identity());
    arc_add_sphere(&w, 1, v3(0, 0, 0), r);
    /* sphere dropped on the step edge: must not stick or jitter */
    arc_add_body(&w, 1.0, arc_inertia_sphere(1.0, r), v3(0.02, -0.75, 0.6), q_identity());
    arc_add_sphere(&w, 2, v3(0, 0, 0), r);
    run(&w, 3.0);
    check_near("seam sphere height", w.bodies[0].x.z, 0.3 + r, 1e-4);
    check_near("seam sphere x drift", w.bodies[0].x.x, 0.5, 1e-4);
    check_near("side sphere on ground", w.bodies[1].x.z, r, 1e-4);
    check(w.bodies[1].x.x <= -r + 1e-4, "side sphere left of the step", w.bodies[1].x.x, -r);
    check_near("edge sphere on plateau", w.bodies[2].x.z, 0.3 + r, 1e-3);
    check_below("edge sphere speed", v3_len(w.bodies[2].v), 1e-3);
}

static void test_set_pose(void)
{
    arc_world w;
    double angles[3] = { 0.3, -0.7, 1.1 };
    int i;
    printf("forward kinematics\n");
    init_empty(&w);
    arc_add_body(&w, 5.0, v3(1, 1, 1), v3(0, 0, 0), q_identity());
    arc_add_body(&w, 1.0, v3(.1, .1, .1), v3(0.2, 0, -0.2), q_identity());
    arc_add_body(&w, 1.0, v3(.1, .1, .1), v3(0.2, 0, -0.6), q_identity());
    arc_add_body(&w, 1.0, v3(.1, .1, .1), v3(0.2, 0.1, -1.0), q_identity());
    arc_add_hinge(&w, 0, 1, v3(0.2, 0, 0), v3(1, 0, 0));
    arc_add_hinge(&w, 1, 2, v3(0.2, 0, -0.4), v3(0, 1, 0));
    arc_add_hinge(&w, 2, 3, v3(0.2, 0, -0.8), v3(0, 0.6, 0.8));
    arc_world_set_pose(&w, v3(1, 2, 3), q_axis_angle(v3(0, 0, 1), 0.8), angles);
    for (i = 0; i < 3; ++i)
        check_near("angle read back", arc_joint_angle(&w, &w.joints[i]), angles[i], 1e-9);
}

/* A blown-up state must be flagged, not crash or invoke undefined behaviour
 * (run under -fsanitize=undefined to check the latter). */
static void test_blow_up(void)
{
    arc_world w;
    double bad[3];
    int i;
    printf("blow-up\n");
    bad[0] = 1e12;
    bad[1] = -1e300;
    bad[2] = sqrt(-1.0); /* NaN */
    for (i = 0; i < 3; ++i) {
        arc_world_init(&w);
        arc_terrain_flat(&w.terrain, 8, 8, 0.5, 0.0);
        arc_add_body(&w, 1.0, arc_inertia_sphere(1.0, 0.1), v3(bad[i], 0.0, 0.05), q_identity());
        arc_add_sphere(&w, 0, v3(0, 0, 0), 0.1);
        arc_world_step(&w, DT, SUBSTEPS);
        check(w.unstable, "flagged unstable", w.unstable, 1);
        check_near("height lookup outside", arc_terrain_height(&w.terrain, bad[i], bad[i]), 0.0, 0.0);
    }
}

int main(void)
{
    test_free_fall();
    test_free_rotation();
    test_pendulum();
    test_chain();
    test_limits();
    test_drive();
    test_resting_sphere();
    test_sliding();
    test_blocks();
    test_set_pose();
    test_blow_up();
    return check_summary();
}
