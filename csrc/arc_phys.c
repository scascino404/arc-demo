/*
 * arc_phys.c - substepped XPBD rigid body solver (C89).
 *
 * Each substep of length h:
 *   1. integrate velocities and positions (gravity, gyroscopic term),
 *   2. solve position constraints once, Gauss-Seidel: joint drives, joint
 *      limits, hinge axis alignment, joint attachment, terrain contacts
 *      (with anchored static and kinetic friction),
 *   3. derive velocities from the position change,
 *   4. solve velocity constraints: drive damping and removal of the normal
 *      contact velocity (restitution 0).
 *
 * Contacts are re-detected every substep. That is cheap here (spheres
 * against a column grid) and avoids stale contact bookkeeping.
 */
#include <string.h>
#include "arc_phys.h"

#define ARC_MAX_SPEED 60.0      /* m/s; beyond this the world is flagged unstable */
#define ARC_MAX_SPIN 400.0      /* rad/s */

/* The solver helpers take and return vec3 by value. Inlined, those stay in
 * registers; as calls they go through stack copies that defeat store-to-load
 * forwarding, which made the solver ~30% slower. */
#if defined(__GNUC__)
#define HOT static __inline__ __attribute__((always_inline))
#else
#define HOT static
#endif

void arc_world_init(arc_world *w)
{
    memset(w, 0, sizeof(*w));
    w->gravity = v3(0.0, 0.0, -9.81);
    w->mu_static = 1.0;
    w->mu_dynamic = 0.8;
    arc_terrain_flat(&w->terrain, 1, 1, 1.0, 0.0);
}

HOT int is_isotropic(vec3 inertia)
{
    return inertia.x == inertia.y && inertia.y == inertia.z;
}

/*
 * q + 0.5 [w, 0] q: applies the small rotation vector w, without
 * renormalizing. Inside a substep the solver lets |q| drift from 1 (the
 * update is linear in q, so the rotation it represents is unaffected, and
 * body_sync scales by 1/|q|^2) and normalizes once before deriving
 * velocities. That keeps a sqrt and a division off the serial chain of
 * constraint solves.
 */
HOT quat q_turn(quat q, vec3 w)
{
    double hx = 0.5 * w.x, hy = 0.5 * w.y, hz = 0.5 * w.z;
    return qt(q.w - hx * q.x - hy * q.y - hz * q.z,
              q.x + hx * q.w + hy * q.z - hz * q.y,
              q.y - hx * q.z + hy * q.w + hz * q.x,
              q.z + hx * q.y - hy * q.x + hz * q.w);
}

/* Refreshes the cached rotation matrix and world-frame inverse inertia from q. */
HOT void body_sync(arc_body *b)
{
    const mat3 *r = &b->rot;
    vec3 d = b->inv_inertia;
    int i, j;
    m3_from_quat(&b->rot, b->q);
    if (is_isotropic(d)) {
        /* rot diag(d) rot^T = d I */
        memset(&b->inv_iw, 0, sizeof(b->inv_iw));
        b->inv_iw.m[0][0] = b->inv_iw.m[1][1] = b->inv_iw.m[2][2] = d.x;
        return;
    }
    for (i = 0; i < 3; ++i) {
        vec3 rd = v3(r->m[i][0] * d.x, r->m[i][1] * d.y, r->m[i][2] * d.z);
        for (j = i; j < 3; ++j)
            b->inv_iw.m[i][j] = b->inv_iw.m[j][i] =
                rd.x * r->m[j][0] + rd.y * r->m[j][1] + rd.z * r->m[j][2];
    }
}

int arc_add_body(arc_world *w, double mass, vec3 inertia, vec3 pos, quat q)
{
    arc_body *b;
    if (w->n_bodies >= ARC_MAX_BODIES)
        return -1;
    b = &w->bodies[w->n_bodies];
    memset(b, 0, sizeof(*b));
    b->x = b->x_prev = pos;
    b->q = b->q_prev = q_normalize(q);
    b->mass = mass;
    b->inv_mass = 1.0 / mass;
    b->inertia = inertia;
    b->inv_inertia = v3(1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z);
    body_sync(b);
    return w->n_bodies++;
}

int arc_add_sphere(arc_world *w, int body, vec3 local_pos, double radius)
{
    arc_sphere *s;
    if (w->n_spheres >= ARC_MAX_SPHERES)
        return -1;
    s = &w->spheres[w->n_spheres];
    memset(s, 0, sizeof(*s));
    s->body = body;
    s->pos = local_pos;
    s->radius = radius;
    return w->n_spheres++;
}

int arc_add_hinge(arc_world *w, int parent, int child, vec3 anchor_world, vec3 axis_world)
{
    arc_joint *j;
    quat qp = q_identity(), qc;
    vec3 xp = v3(0.0, 0.0, 0.0), axis, ref;
    if (w->n_joints >= ARC_MAX_JOINTS || child <= parent)
        return -1;
    if (parent >= 0) {
        xp = w->bodies[parent].x;
        qp = w->bodies[parent].q;
    }
    qc = w->bodies[child].q;
    axis = v3_normalize(axis_world);
    ref = v3_perp(axis);

    j = &w->joints[w->n_joints];
    memset(j, 0, sizeof(*j));
    j->parent = parent;
    j->child = child;
    j->r1 = q_inv_rotate(qp, v3_sub(anchor_world, xp));
    j->r2 = q_inv_rotate(qc, v3_sub(anchor_world, w->bodies[child].x));
    j->a1 = q_inv_rotate(qp, axis);
    j->a2 = q_inv_rotate(qc, axis);
    j->b1 = q_inv_rotate(qp, ref);
    j->b2 = q_inv_rotate(qc, ref);
    j->q_rel0 = q_mul(q_conj(qp), qc);
    j->lo = 1.0;
    j->hi = -1.0;
    return w->n_joints++;
}

int arc_add_visual(arc_world *w, int type, int body, vec3 p0, vec3 p1,
                   double radius, double radius2, int material, float r, float g, float b)
{
    arc_visual *v;
    if (w->n_visuals >= ARC_MAX_VISUALS)
        return -1;
    v = &w->visuals[w->n_visuals];
    v->type = type;
    v->body = body;
    v->p0 = p0;
    v->p1 = p1;
    v->radius = radius;
    v->radius2 = radius2;
    v->aspect = 1.0;
    v->material = material;
    v->color[0] = r;
    v->color[1] = g;
    v->color[2] = b;
    return w->n_visuals++;
}

/* ------------------------------------------------------------------------ */
/* Constraint primitives */

/* I^-1 v, with v and the result in world frame. */
HOT vec3 inv_inertia_world(const arc_body *b, vec3 v)
{
    return m3_mul_v3(&b->inv_iw, v);
}

/* Body-frame vector v in world frame; b == NULL is the static world. */
HOT vec3 body_rotate(const arc_body *b, vec3 v)
{
    return b ? m3_mul_v3(&b->rot, v) : v;
}

/* Generalized inverse mass of a body for a unit impulse n applied at offset r. */
HOT double w_pos(const arc_body *b, vec3 r, vec3 n)
{
    vec3 rn;
    if (!b)
        return 0.0;
    rn = v3_cross(r, n);
    return b->inv_mass + v3_dot(rn, inv_inertia_world(b, rn));
}

HOT double w_rot(const arc_body *b, vec3 n)
{
    return b ? v3_dot(n, inv_inertia_world(b, n)) : 0.0;
}

HOT void apply_pos_impulse(arc_body *b, vec3 r, vec3 p)
{
    if (!b)
        return;
    b->x = v3_madd(b->x, p, b->inv_mass);
    b->q = q_turn(b->q, inv_inertia_world(b, v3_cross(r, p)));
    body_sync(b);
}

HOT void apply_rot_impulse(arc_body *b, vec3 p)
{
    if (!b)
        return;
    b->q = q_turn(b->q, inv_inertia_world(b, p));
    body_sync(b);
}

HOT void apply_vel_impulse(arc_body *b, vec3 r, vec3 p)
{
    if (!b)
        return;
    b->v = v3_madd(b->v, p, b->inv_mass);
    b->w = v3_add(b->w, inv_inertia_world(b, v3_cross(r, p)));
}

/*
 * Positional constraint: move the point at offset r1 of b1 by d relative to
 * the point at offset r2 of b2 (either body may be NULL for the static
 * world). alpha is the compliance. Returns the impulse magnitude.
 */
HOT double solve_pos(arc_body *b1, vec3 r1, arc_body *b2, vec3 r2,
                     vec3 d, double alpha, double h)
{
    double c = v3_len(d), denom, dl;
    vec3 n, p;
    if (c < 1e-12)
        return 0.0;
    n = v3_scale(d, 1.0 / c);
    denom = w_pos(b1, r1, n) + w_pos(b2, r2, n) + alpha / (h * h);
    if (denom <= 0.0)
        return 0.0;
    dl = c / denom;
    p = v3_scale(n, dl);
    apply_pos_impulse(b1, r1, p);
    apply_pos_impulse(b2, r2, v3_neg(p));
    return dl;
}

/* Solves the 3x3 system K x = d (Cramer's rule). Returns 0 if singular. */
static int solve3(const double K[3][3], vec3 d, vec3 *x)
{
    double c00 = K[1][1] * K[2][2] - K[1][2] * K[2][1];
    double c01 = K[1][2] * K[2][0] - K[1][0] * K[2][2];
    double c02 = K[1][0] * K[2][1] - K[1][1] * K[2][0];
    double det = K[0][0] * c00 + K[0][1] * c01 + K[0][2] * c02, inv;
    vec3 c0, c1, c2;
    if (fabs(det) < 1e-300)
        return 0;
    inv = 1.0 / det;
    c0 = v3(K[0][0], K[1][0], K[2][0]);
    c1 = v3(K[0][1], K[1][1], K[2][1]);
    c2 = v3(K[0][2], K[1][2], K[2][2]);
    x->x = v3_dot(d, v3_cross(c1, c2)) * inv;
    x->y = v3_dot(c0, v3_cross(d, c2)) * inv;
    x->z = v3_dot(c0, v3_cross(c1, d)) * inv;
    return 1;
}

/* Solves K x = d for a symmetric K (adjugate). Returns 0 if singular. */
HOT int solve3_sym(const double K[3][3], vec3 d, vec3 *x)
{
    double a = K[0][0], b = K[0][1], c = K[0][2], e = K[1][1], f = K[1][2], g = K[2][2];
    double A = e * g - f * f, B = c * f - b * g, C = b * f - c * e;
    double det = a * A + b * B + c * C, inv, D, E, F;
    if (fabs(det) < 1e-300)
        return 0;
    inv = 1.0 / det;
    D = a * g - c * c;
    E = b * c - a * f;
    F = a * e - b * b;
    x->x = (A * d.x + B * d.y + C * d.z) * inv;
    x->y = (B * d.x + D * d.y + E * d.z) * inv;
    x->z = (C * d.x + E * d.y + F * d.z) * inv;
    return 1;
}

/* K += point mass matrix of body b at offset r: dv = K p for an impulse p,
 * K = I / m - [r]x I_w^-1 [r]x, written out (it is symmetric). */
HOT void add_point_mass(double K[3][3], const arc_body *b, vec3 r)
{
    const mat3 *a;
    double x = r.x, y = r.y, z = r.z, k01, k02, k12;
    if (!b)
        return;
    a = &b->inv_iw;
    K[0][0] += b->inv_mass + a->m[1][1] * z * z - 2.0 * a->m[1][2] * y * z + a->m[2][2] * y * y;
    K[1][1] += b->inv_mass + a->m[2][2] * x * x - 2.0 * a->m[0][2] * x * z + a->m[0][0] * z * z;
    K[2][2] += b->inv_mass + a->m[0][0] * y * y - 2.0 * a->m[0][1] * x * y + a->m[1][1] * x * x;
    k01 = a->m[1][2] * x * z + a->m[0][2] * y * z - a->m[0][1] * z * z - a->m[2][2] * x * y;
    k02 = a->m[0][1] * y * z + a->m[1][2] * x * y - a->m[1][1] * x * z - a->m[0][2] * y * y;
    k12 = a->m[0][2] * x * y + a->m[0][1] * x * z - a->m[0][0] * y * z - a->m[1][2] * x * x;
    K[0][1] += k01; K[1][0] += k01;
    K[0][2] += k02; K[2][0] += k02;
    K[1][2] += k12; K[2][1] += k12;
}

/*
 * Rigid point-to-point constraint solved as a 3x3 block, so the coupling
 * between directions is exact to first order. A one-direction solve
 * (solve_pos) converges badly for joints whose anchor is far from a body
 * with small inertia, e.g. a point mass pendulum.
 */
HOT void solve_point(arc_body *b1, vec3 r1, arc_body *b2, vec3 r2, vec3 d)
{
    double K[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    vec3 p;
    if (v3_len2(d) < 1e-24)
        return;
    add_point_mass(K, b1, r1);
    add_point_mass(K, b2, r2);
    if (!solve3_sym((const double (*)[3])K, d, &p))
        return;
    apply_pos_impulse(b1, r1, p);
    apply_pos_impulse(b2, r2, v3_neg(p));
}

/*
 * Rigid angular constraint solved as a 3x3 block: afterwards b1 has rotated
 * by exactly rot relative to b2 (to first order). A one-direction solve
 * leaks rotation into other axes when the world inertia is anisotropic,
 * e.g. spinning a slender leg segment about its own axis.
 */
HOT void solve_rot_block(arc_body *b1, arc_body *b2, vec3 rot)
{
    double K[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    vec3 p;
    int i, k;
    if (v3_len2(rot) < 1e-24)
        return;
    for (i = 0; i < 3; ++i)
        for (k = 0; k < 3; ++k) {
            if (b1)
                K[i][k] += b1->inv_iw.m[i][k];
            if (b2)
                K[i][k] += b2->inv_iw.m[i][k];
        }
    if (!solve3_sym((const double (*)[3])K, rot, &p))
        return;
    apply_rot_impulse(b1, p);
    apply_rot_impulse(b2, v3_neg(p));
}

/*
 * Angular constraint: rotate b1 by the rotation vector rot relative to b2.
 * The impulse is capped at max_dl. Returns the applied impulse vector
 * (acting on b1; b2 receives its opposite).
 */
HOT vec3 solve_rot(arc_body *b1, arc_body *b2, vec3 rot, double alpha,
                   double h, double max_dl)
{
    double theta = v3_len(rot), denom, dl;
    vec3 n, p;
    if (theta < 1e-12)
        return v3(0.0, 0.0, 0.0);
    n = v3_scale(rot, 1.0 / theta);
    denom = w_rot(b1, n) + w_rot(b2, n) + alpha / (h * h);
    if (denom <= 0.0)
        return v3(0.0, 0.0, 0.0);
    dl = theta / denom;
    if (dl > max_dl)
        dl = max_dl;
    p = v3_scale(n, dl);
    apply_rot_impulse(b1, p);
    apply_rot_impulse(b2, v3_neg(p));
    return p;
}

/* ------------------------------------------------------------------------ */
/* Joints */

static arc_body *joint_parent(arc_world *w, const arc_joint *j)
{
    return j->parent >= 0 ? &w->bodies[j->parent] : 0;
}

static quat body_q(const arc_world *w, int i)
{
    return i >= 0 ? w->bodies[i].q : q_identity();
}

double arc_joint_angle(const arc_world *w, const arc_joint *j)
{
    quat qp = body_q(w, j->parent), qc = body_q(w, j->child);
    vec3 a = q_rotate(qp, j->a1);
    vec3 b1 = q_rotate(qp, j->b1);
    vec3 b2 = q_rotate(qc, j->b2);
    return atan2(v3_dot(v3_cross(b1, b2), a), v3_dot(b1, b2));
}

double arc_joint_rate(const arc_world *w, const arc_joint *j)
{
    vec3 a = q_rotate(body_q(w, j->parent), j->a1);
    vec3 wp = j->parent >= 0 ? w->bodies[j->parent].w : v3(0.0, 0.0, 0.0);
    return v3_dot(v3_sub(w->bodies[j->child].w, wp), a);
}

/* Sine and cosine of the joint angle (as in arc_joint_angle) times a common
 * positive factor, from the cached body rotations; a1 is the world axis. */
HOT void joint_sincos(const arc_body *bp, const arc_body *bc, const arc_joint *j,
                      vec3 a1, double *s, double *c)
{
    vec3 b1 = body_rotate(bp, j->b1);
    vec3 b2 = body_rotate(bc, j->b2);
    *s = v3_dot(v3_cross(b1, b2), a1);
    *c = v3_dot(b1, b2);
}

static void solve_joint_positions(arc_world *w, arc_joint *j, double h)
{
    arc_body *bp = joint_parent(w, j);
    arc_body *bc = &w->bodies[j->child];
    vec3 a1, r1, r2, p1, p2;
    double phi, s, c;

    /* PD drive: compliant, torque limited rotation towards the target. The
     * parent is rotated by (phi - target) relative to the child. */
    if (j->kp > 0.0) {
        vec3 p;
        a1 = body_rotate(bp, j->a1);
        joint_sincos(bp, bc, j, a1, &s, &c);
        phi = atan2(s, c);
        p = solve_rot(bp, bc, v3_scale(a1, phi - j->target), 1.0 / j->kp, h,
                      j->max_torque * h * h);
        j->torque_acc -= v3_dot(p, a1) / (h * h);
    }

    /* limits. The angle is only needed outside [lo, hi]: for a range of at
     * most pi, sin(phi - lo) >= 0 and sin(hi - phi) >= 0 mean inside. */
    if (j->lo <= j->hi) {
        a1 = body_rotate(bp, j->a1);
        joint_sincos(bp, bc, j, a1, &s, &c);
        if (j->hi - j->lo > ARC_PI || s * j->lim[0] - c * j->lim[1] < 0.0
            || j->lim[3] * c - j->lim[2] * s < 0.0) {
            phi = atan2(s, c);
            if (phi < j->lo)
                solve_rot_block(bp, bc, v3_scale(a1, phi - j->lo));
            else if (phi > j->hi)
                solve_rot_block(bp, bc, v3_scale(a1, phi - j->hi));
        }
    }

    /* hinge axes alignment */
    a1 = body_rotate(bp, j->a1);
    solve_rot_block(bp, bc, v3_cross(a1, body_rotate(bc, j->a2)));

    /* attachment */
    r1 = body_rotate(bp, j->r1);
    r2 = body_rotate(bc, j->r2);
    p1 = bp ? v3_add(bp->x, r1) : r1;
    p2 = v3_add(bc->x, r2);
    solve_point(bp, r1, bc, r2, v3_sub(p2, p1));
}

static void solve_joint_velocities(arc_world *w, arc_joint *j, double h)
{
    arc_body *bp = joint_parent(w, j);
    arc_body *bc = &w->bodies[j->child];
    vec3 a, wp;
    double rel, wsum, impulse, cap;
    if (j->kd <= 0.0)
        return;
    a = body_rotate(bp, j->a1);
    wp = bp ? bp->w : v3(0.0, 0.0, 0.0);
    rel = v3_dot(v3_sub(bc->w, wp), a);
    wsum = w_rot(bp, a) + w_rot(bc, a);
    if (wsum <= 0.0)
        return;
    /* damping torque -kd * rel, never reversing the relative motion */
    impulse = -j->kd * rel * h;
    cap = arc_min(fabs(rel) / wsum, j->max_torque * h);
    impulse = arc_clamp(impulse, -cap, cap);
    if (bp)
        bp->w = v3_sub(bp->w, inv_inertia_world(bp, v3_scale(a, impulse)));
    bc->w = v3_add(bc->w, inv_inertia_world(bc, v3_scale(a, impulse)));
    j->torque_acc += impulse / h;
}

/* ------------------------------------------------------------------------ */
/* Contacts */

/* Normal contact: push the point out of the terrain and record the contact
 * for the velocity pass. Returns the normal impulse. */
HOT double solve_contact(arc_world *w, int bi, vec3 point, vec3 n, double depth, double h)
{
    arc_body *b = &w->bodies[bi];
    vec3 r = v3_sub(point, b->x);
    double ln = solve_pos(b, r, 0, r, v3_scale(n, depth), 0.0, h);
    if (w->n_contacts < ARC_MAX_CONTACTS) {
        arc_contact *c = &w->contacts[w->n_contacts++];
        c->body = bi;
        c->local = m3_tmul_v3(&b->rot, r);
        c->n = n;
    }
    return ln;
}

/*
 * Friction at position level with a persistent anchor per sphere: the
 * ground point where the sphere touched down. The tangential offset of the
 * current contact point from the anchor is corrected when mu_s * normal
 * impulse can hold it; otherwise the correction is clamped to mu_d * normal
 * impulse (Macklin et al. 2014) and the anchor slides along. Undoing only
 * the motion of the current substep (instead of returning to the anchor)
 * lets feet creep when the joints keep pushing them sideways, and lets
 * spheres roll away.
 */
HOT void solve_friction(arc_world *w, arc_sphere *s, vec3 n, double ln, double h)
{
    arc_body *b = &w->bodies[s->body];
    vec3 c = v3_add(b->x, body_rotate(b, s->pos));
    vec3 p = v3_madd(c, n, -s->radius), e, r;
    double len, wt, lt, frac = 1.0;

    if (!s->anchored || v3_dot(n, s->anchor_n) < 0.9) {
        /* touch down: anchor where the contact point was at the substep start */
        vec3 c0 = v3_add(b->x_prev, q_rotate(b->q_prev, s->pos));
        s->anchor = v3_madd(c0, n, -s->radius);
        s->anchor_n = n;
        s->anchored = 1;
    }
    e = v3_sub(p, s->anchor);
    e = v3_madd(e, n, -v3_dot(e, n));
    len = v3_len(e);
    if (len < 1e-12)
        return;
    r = v3_sub(p, b->x);
    wt = w_pos(b, r, v3_scale(e, 1.0 / len));
    lt = len / wt;
    if (lt > w->mu_static * ln) {
        frac = arc_min(w->mu_dynamic * ln / lt, 1.0);
        s->anchor = v3_madd(s->anchor, e, 1.0 - frac);
    }
    solve_pos(b, r, 0, r, v3_scale(e, -frac), 0.0, h);
}

/* Collides one sphere with the column grid, solving each contact at once. */
static void collide_sphere(arc_world *w, arc_sphere *s, double h)
{
    const arc_terrain *t = &w->terrain;
    arc_body *b = &w->bodies[s->body];
    double rad = s->radius, ln = 0.0;
    vec3 c = v3_add(b->x, body_rotate(b, s->pos)), n_max = v3(0.0, 0.0, 1.0);
    double depth_max = 0.0;
    int ix0 = (int)floor((c.x - rad - t->ox) / t->cell);
    int ix1 = (int)floor((c.x + rad - t->ox) / t->cell);
    int iy0 = (int)floor((c.y - rad - t->oy) / t->cell);
    int iy1 = (int)floor((c.y + rad - t->oy) / t->cell);
    int ix, iy, sx, sy, touched = 0;

    for (iy = iy0; iy <= iy1; ++iy) {
        for (ix = ix0; ix <= ix1; ++ix) {
            double top = arc_terrain_cell(t, ix, iy);
            double x0 = t->ox + ix * t->cell, x1 = x0 + t->cell;
            double y0 = t->oy + iy * t->cell, y1 = y0 + t->cell;
            double dist, depth;
            vec3 q, diff, n;

            if (c.z - rad >= top)
                continue;
            /* Skip edges and corners that are buried by an equal or higher
             * neighbour: the neighbour column produces the right contact. */
            sx = c.x > x1 ? 1 : (c.x < x0 ? -1 : 0);
            sy = c.y > y1 ? 1 : (c.y < y0 ? -1 : 0);
            if (sx && arc_terrain_cell(t, ix + sx, iy) >= top)
                continue;
            if (sy && arc_terrain_cell(t, ix, iy + sy) >= top)
                continue;
            if (sx && sy && arc_terrain_cell(t, ix + sx, iy + sy) >= top)
                continue;

            q = v3(arc_clamp(c.x, x0, x1), arc_clamp(c.y, y0, y1), arc_min(c.z, top));
            diff = v3_sub(c, q);
            dist = v3_len(diff);
            if (dist > 1e-9) {
                if (dist >= rad)
                    continue;
                n = v3_scale(diff, 1.0 / dist);
                depth = rad - dist;
            } else {
                /* center inside the column: push out through the top */
                n = v3(0.0, 0.0, 1.0);
                depth = top - c.z + rad;
            }
            ln += solve_contact(w, s->body, v3_madd(c, n, -rad), n, depth, h);
            c = v3_add(b->x, body_rotate(b, s->pos));
            if (depth > depth_max) {
                depth_max = depth;
                n_max = n;
            }
            touched = 1;
        }
    }
    if (touched)
        solve_friction(w, s, n_max, ln, h);
    else
        s->anchored = 0;
    s->touch += touched;
}

static void solve_contact_velocities(arc_world *w)
{
    int i;
    for (i = 0; i < w->n_contacts; ++i) {
        arc_contact *c = &w->contacts[i];
        arc_body *b = &w->bodies[c->body];
        vec3 r = body_rotate(b, c->local);
        vec3 vc = v3_add(b->v, v3_cross(b->w, r));
        double vn = v3_dot(c->n, vc);

        /* restitution 0: remove the normal velocity, including the
         * separating velocity introduced by the position correction */
        apply_vel_impulse(b, r, v3_scale(c->n, -vn / w_pos(b, r, c->n)));
    }
}

/* ------------------------------------------------------------------------ */
/* Stepping */

/*
 * Implicit gyroscopic update in body frame (one Newton step, Catto 2015):
 * solve I (w1 - w0) + h w1 x (I w1) = 0. Unlike the explicit update it does
 * not gain energy for fast spinning bodies.
 */
static vec3 gyroscopic(const arc_body *b, vec3 w0, double h)
{
    double J[3][3];
    vec3 iw = v3_mul(w0, b->inertia), f = v3_scale(v3_cross(w0, iw), h), dw;
    double in[3];
    int k;
    in[0] = b->inertia.x;
    in[1] = b->inertia.y;
    in[2] = b->inertia.z;
    for (k = 0; k < 3; ++k) {
        vec3 e = v3(k == 0, k == 1, k == 2), col;
        col = v3_add(v3_scale(e, in[k]),
                     v3_scale(v3_sub(v3_scale(v3_cross(w0, e), in[k]), v3_cross(iw, e)), h));
        J[0][k] = col.x;
        J[1][k] = col.y;
        J[2][k] = col.z;
    }
    if (!solve3((const double (*)[3])J, f, &dw))
        return w0;
    return v3_sub(w0, dw);
}

static void integrate(arc_world *w, double h)
{
    int i;
    for (i = 0; i < w->n_bodies; ++i) {
        arc_body *b = &w->bodies[i];
        b->x_prev = b->x;
        b->q_prev = b->q;
        b->v = v3_madd(b->v, w->gravity, h);
        b->x = v3_madd(b->x, b->v, h);
        if (!is_isotropic(b->inertia)) /* w x (I w) = 0 */
            b->w = m3_mul_v3(&b->rot, gyroscopic(b, m3_tmul_v3(&b->rot, b->w), h));
        b->q = q_turn(b->q, v3_scale(b->w, h));
        body_sync(b);
    }
}

static void update_velocities(arc_world *w, double h)
{
    int i;
    for (i = 0; i < w->n_bodies; ++i) {
        arc_body *b = &w->bodies[i];
        quat dq;
        b->q = q_normalize(b->q); /* see q_turn */
        dq = q_mul(b->q, q_conj(b->q_prev));
        b->v = v3_scale(v3_sub(b->x, b->x_prev), 1.0 / h);
        b->w = v3_scale(v3(dq.x, dq.y, dq.z), 2.0 / h);
        if (dq.w < 0.0)
            b->w = v3_neg(b->w);
    }
}

static int is_finite(double v)
{
    return v == v && v < 1e300 && v > -1e300;
}

static void check_stability(arc_world *w)
{
    int i;
    for (i = 0; i < w->n_bodies; ++i) {
        arc_body *b = &w->bodies[i];
        double v2 = v3_len2(b->v), w2 = v3_len2(b->w);
        if (!is_finite(v2) || !is_finite(w2) || !is_finite(b->x.x + b->x.y + b->x.z)
            || v2 > ARC_MAX_SPEED * ARC_MAX_SPEED || w2 > ARC_MAX_SPIN * ARC_MAX_SPIN)
            w->unstable = 1;
    }
}

void arc_world_step(arc_world *w, double dt, int substeps)
{
    double h = dt / substeps;
    int s, i;

    /* the caller may have changed orientations since the last step */
    for (i = 0; i < w->n_bodies; ++i)
        body_sync(&w->bodies[i]);
    for (i = 0; i < w->n_spheres; ++i)
        w->spheres[i].touch = 0;
    for (i = 0; i < w->n_joints; ++i) {
        arc_joint *j = &w->joints[i];
        j->torque_acc = 0.0;
        j->lim[0] = cos(j->lo);
        j->lim[1] = sin(j->lo);
        j->lim[2] = cos(j->hi);
        j->lim[3] = sin(j->hi);
    }

    for (s = 0; s < substeps; ++s) {
        integrate(w, h);
        w->n_contacts = 0;
        for (i = 0; i < w->n_joints; ++i)
            solve_joint_positions(w, &w->joints[i], h);
        for (i = 0; i < w->n_spheres; ++i)
            collide_sphere(w, &w->spheres[i], h);
        update_velocities(w, h);
        for (i = 0; i < w->n_joints; ++i)
            solve_joint_velocities(w, &w->joints[i], h);
        solve_contact_velocities(w);
    }

    for (i = 0; i < w->n_joints; ++i) {
        arc_joint *j = &w->joints[i];
        j->angle = arc_joint_angle(w, j);
        j->rate = arc_joint_rate(w, j);
        j->torque = j->torque_acc / substeps;
    }
    w->time += dt;
    check_stability(w);
}

void arc_world_set_pose(arc_world *w, vec3 pos, quat q, const double *angles)
{
    int i;
    arc_body *root = &w->bodies[0];
    root->x = pos;
    root->q = q_normalize(q);
    for (i = 0; i < w->n_joints; ++i) {
        arc_joint *j = &w->joints[i];
        arc_body *c = &w->bodies[j->child];
        vec3 xp = j->parent >= 0 ? w->bodies[j->parent].x : v3(0.0, 0.0, 0.0);
        quat qp = body_q(w, j->parent);
        vec3 anchor = v3_add(xp, q_rotate(qp, j->r1));
        c->q = q_normalize(q_mul(qp, q_mul(q_axis_angle(j->a1, angles[i]), j->q_rel0)));
        c->x = v3_sub(anchor, q_rotate(c->q, j->r2));
        j->target = angles[i];
        j->angle = angles[i];
        j->rate = 0.0;
        j->torque = 0.0;
    }
    for (i = 0; i < w->n_bodies; ++i) {
        arc_body *b = &w->bodies[i];
        b->x_prev = b->x;
        b->q_prev = b->q;
        b->v = v3(0.0, 0.0, 0.0);
        b->w = v3(0.0, 0.0, 0.0);
        body_sync(b);
    }
    for (i = 0; i < w->n_spheres; ++i) {
        w->spheres[i].touch = 0;
        w->spheres[i].anchored = 0;
    }
    w->n_contacts = 0;
    w->unstable = 0;
}

double arc_world_energy(const arc_world *w)
{
    double e = 0.0;
    int i;
    for (i = 0; i < w->n_bodies; ++i) {
        const arc_body *b = &w->bodies[i];
        vec3 wl = q_inv_rotate(b->q, b->w);
        e += 0.5 * b->mass * v3_len2(b->v);
        e += 0.5 * v3_dot(wl, v3_mul(wl, b->inertia));
        e -= b->mass * v3_dot(w->gravity, b->x);
    }
    return e;
}

/* ------------------------------------------------------------------------ */
/* Inertia */

vec3 arc_inertia_box(double mass, vec3 he)
{
    double k = mass / 3.0;
    return v3(k * (he.y * he.y + he.z * he.z),
              k * (he.x * he.x + he.z * he.z),
              k * (he.x * he.x + he.y * he.y));
}

vec3 arc_inertia_capsule_z(double mass, double radius, double half_length)
{
    /* approximated as a solid cylinder spanning the whole capsule */
    double len = 2.0 * (half_length + radius);
    double ixx = mass * (3.0 * radius * radius + len * len) / 12.0;
    return v3(ixx, ixx, 0.5 * mass * radius * radius);
}

vec3 arc_inertia_sphere(double mass, double radius)
{
    double i = 0.4 * mass * radius * radius;
    return v3(i, i, i);
}
