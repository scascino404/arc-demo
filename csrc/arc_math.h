/*
 * arc_math.h - small vector / quaternion library (C89).
 *
 * Conventions: right-handed, Z up, doubles everywhere. Quaternions are
 * unit quaternions (w, x, y, z) rotating body-local vectors into world space.
 */
#ifndef ARC_MATH_H
#define ARC_MATH_H

#include <math.h>

#if defined(__GNUC__)
#define ARC_INLINE static __inline__
#else
#define ARC_INLINE static
#endif

#define ARC_PI 3.14159265358979323846

typedef struct { double x, y, z; } vec3;
typedef struct { double w, x, y, z; } quat;
typedef struct { double m[3][3]; } mat3;   /* row-major */

ARC_INLINE vec3 v3(double x, double y, double z)
{
    vec3 r;
    r.x = x; r.y = y; r.z = z;
    return r;
}

ARC_INLINE vec3 v3_add(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
ARC_INLINE vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
ARC_INLINE vec3 v3_scale(vec3 a, double s) { return v3(a.x * s, a.y * s, a.z * s); }
ARC_INLINE vec3 v3_neg(vec3 a) { return v3(-a.x, -a.y, -a.z); }
ARC_INLINE vec3 v3_mul(vec3 a, vec3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }

/* a + b * s */
ARC_INLINE vec3 v3_madd(vec3 a, vec3 b, double s)
{
    return v3(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s);
}

ARC_INLINE double v3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

ARC_INLINE vec3 v3_cross(vec3 a, vec3 b)
{
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

ARC_INLINE double v3_len2(vec3 a) { return v3_dot(a, a); }
ARC_INLINE double v3_len(vec3 a) { return sqrt(v3_dot(a, a)); }

/* Returns the zero vector when a is (numerically) zero. */
ARC_INLINE vec3 v3_normalize(vec3 a)
{
    double l = v3_len(a);
    return l > 1e-15 ? v3_scale(a, 1.0 / l) : v3(0.0, 0.0, 0.0);
}

/* Any unit vector perpendicular to the unit vector n. */
ARC_INLINE vec3 v3_perp(vec3 n)
{
    vec3 t = fabs(n.x) < 0.57 ? v3(1.0, 0.0, 0.0) : v3(0.0, 1.0, 0.0);
    return v3_normalize(v3_cross(n, t));
}

ARC_INLINE quat qt(double w, double x, double y, double z)
{
    quat r;
    r.w = w; r.x = x; r.y = y; r.z = z;
    return r;
}

ARC_INLINE quat q_identity(void) { return qt(1.0, 0.0, 0.0, 0.0); }
ARC_INLINE quat q_conj(quat a) { return qt(a.w, -a.x, -a.y, -a.z); }

ARC_INLINE quat q_mul(quat a, quat b)
{
    return qt(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
              a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
              a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
              a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

ARC_INLINE quat q_normalize(quat a)
{
    double l = sqrt(a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z);
    if (l < 1e-15)
        return q_identity();
    l = 1.0 / l;
    return qt(a.w * l, a.x * l, a.y * l, a.z * l);
}

/* Rotation of `angle` radians about the unit vector `axis`. */
ARC_INLINE quat q_axis_angle(vec3 axis, double angle)
{
    double s = sin(0.5 * angle);
    return qt(cos(0.5 * angle), axis.x * s, axis.y * s, axis.z * s);
}

ARC_INLINE vec3 q_rotate(quat q, vec3 v)
{
    /* v + 2w (u x v) + 2 u x (u x v), with u the vector part of q */
    vec3 u = v3(q.x, q.y, q.z);
    vec3 t = v3_scale(v3_cross(u, v), 2.0);
    return v3_add(v3_madd(v, t, q.w), v3_cross(u, t));
}

ARC_INLINE vec3 q_inv_rotate(quat q, vec3 v) { return q_rotate(q_conj(q), v); }

/* Rotation matrix of q / |q| (q need not be normalized, only nonzero). */
ARC_INLINE void m3_from_quat(mat3 *r, quat q)
{
    double k = 2.0 / (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    double x2 = q.x * k, y2 = q.y * k, z2 = q.z * k;
    double xx = q.x * x2, yy = q.y * y2, zz = q.z * z2;
    double xy = q.x * y2, xz = q.x * z2, yz = q.y * z2;
    double wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
    r->m[0][0] = 1.0 - (yy + zz); r->m[0][1] = xy - wz;         r->m[0][2] = xz + wy;
    r->m[1][0] = xy + wz;         r->m[1][1] = 1.0 - (xx + zz); r->m[1][2] = yz - wx;
    r->m[2][0] = xz - wy;         r->m[2][1] = yz + wx;         r->m[2][2] = 1.0 - (xx + yy);
}

ARC_INLINE vec3 m3_mul_v3(const mat3 *a, vec3 v)
{
    return v3(a->m[0][0] * v.x + a->m[0][1] * v.y + a->m[0][2] * v.z,
              a->m[1][0] * v.x + a->m[1][1] * v.y + a->m[1][2] * v.z,
              a->m[2][0] * v.x + a->m[2][1] * v.y + a->m[2][2] * v.z);
}

/* a^T v */
ARC_INLINE vec3 m3_tmul_v3(const mat3 *a, vec3 v)
{
    return v3(a->m[0][0] * v.x + a->m[1][0] * v.y + a->m[2][0] * v.z,
              a->m[0][1] * v.x + a->m[1][1] * v.y + a->m[2][1] * v.z,
              a->m[0][2] * v.x + a->m[1][2] * v.y + a->m[2][2] * v.z);
}

ARC_INLINE double arc_clamp(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

ARC_INLINE double arc_min(double a, double b) { return a < b ? a : b; }
ARC_INLINE double arc_max(double a, double b) { return a > b ? a : b; }

#endif
