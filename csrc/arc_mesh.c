/*
 * arc_mesh.c - scene geometry for the renderer, in immediate mode OpenGL:
 * positions, normals, and for the terrain vertex colors.
 */
#include <GL/gl.h>
#include "arc_mesh.h"

static void vertex(vec3 p, vec3 n)
{
    glNormal3d(n.x, n.y, n.z);
    glVertex3d(p.x, p.y, p.z);
}

void arc_mesh_sphere(void)
{
    int i, j, slices = 48, stacks = 32;
    for (i = 0; i < stacks; ++i) {
        double t0 = ARC_PI * i / stacks - 0.5 * ARC_PI;
        double t1 = ARC_PI * (i + 1) / stacks - 0.5 * ARC_PI;
        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= slices; ++j) {
            double p = 2.0 * ARC_PI * j / slices;
            vec3 a = v3(cos(t1) * cos(p), cos(t1) * sin(p), sin(t1));
            vec3 b = v3(cos(t0) * cos(p), cos(t0) * sin(p), sin(t0));
            vertex(a, a);
            vertex(b, b);
        }
        glEnd();
    }
}

/* Rotation taking z onto the unit vector d, as a body-frame basis. */
static void basis(vec3 d, vec3 *u, vec3 *v)
{
    *u = v3_normalize(fabs(d.z) < 0.9 ? v3_cross(v3(0, 0, 1), d) : v3_cross(v3(1, 0, 0), d));
    *v = v3_cross(d, *u);
}

/* Frame along p0 -> p1 that keeps the body axes when the segment runs
 * along body z (the legs), so cross sections line up with body x / y. */
static void segment_frame(vec3 p0, vec3 p1, vec3 *d, vec3 *u, vec3 *v, double *len)
{
    vec3 s = v3_sub(p1, p0);
    *len = v3_len(s);
    *d = v3_scale(s, 1.0 / (*len > 1e-12 ? *len : 1.0));
    if (fabs(d->x) < 1e-9 && fabs(d->y) < 1e-9) {
        *u = v3(1, 0, 0);
        *v = v3(0, d->z > 0 ? 1 : -1, 0);
    } else {
        basis(*d, u, v);
    }
}

static vec3 frame_point(vec3 o, vec3 u, vec3 v, vec3 d, double x, double y, double z)
{
    return v3_add(o, v3_add(v3_scale(u, x), v3_add(v3_scale(v, y), v3_scale(d, z))));
}

/* Disc cap of a cylinder at distance z along d, facing nd. */
static void cap(vec3 o, vec3 u, vec3 v, vec3 d, double z, double rad, double nd, int slices)
{
    int j;
    glBegin(GL_TRIANGLE_FAN);
    vertex(v3_madd(o, d, z), v3_scale(d, nd));
    for (j = 0; j <= slices; ++j) {
        double p = 2.0 * ARC_PI * j / slices;
        vertex(frame_point(o, u, v, d, rad * cos(p), rad * sin(p), z), v3_scale(d, nd));
    }
    glEnd();
}

void arc_mesh_cylinder(vec3 p0, vec3 p1, double rad, int ribbed)
{
    vec3 d, u, v;
    double len, pitch = 0.010, depth = ribbed ? 0.12 * rad : 0.0;
    int i, j, slices = ribbed ? 16 : 36, rings;
    segment_frame(p0, p1, &d, &u, &v, &len);
    if (len < 1e-9)
        return;
    rings = ribbed ? (int)ceil(len / pitch) * 4 : 1;
    for (i = 0; i < rings; ++i) {
        double z0 = len * i / rings, z1 = len * (i + 1) / rings;
        double k = 2.0 * ARC_PI / pitch;
        double r0 = rad - depth * (0.5 - 0.5 * cos(k * z0)), r1 = rad - depth * (0.5 - 0.5 * cos(k * z1));
        double s0 = depth * 0.5 * k * sin(k * z0), s1 = depth * 0.5 * k * sin(k * z1); /* -dr/dz */
        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= slices; ++j) {
            double p = 2.0 * ARC_PI * j / slices, c = cos(p), s = sin(p);
            vec3 radial = v3_add(v3_scale(u, c), v3_scale(v, s));
            vertex(frame_point(p0, u, v, d, r1 * c, r1 * s, z1), v3_normalize(v3_madd(radial, d, s1)));
            vertex(frame_point(p0, u, v, d, r0 * c, r0 * s, z0), v3_normalize(v3_madd(radial, d, s0)));
        }
        glEnd();
    }
    cap(p0, u, v, d, 0.0, rad, -1.0, slices);
    cap(p0, u, v, d, len, rad, 1.0, slices);
}

void arc_mesh_taper(vec3 p0, vec3 p1, double r0, double r1, double aspect)
{
    static const double PX[8] = { 1, 0.6, -0.6, -1, -1, -0.6, 0.6, 1 };
    static const double PY[8] = { 0.45, 1, 1, 0.45, -0.45, -1, -1, -0.45 };
    vec3 d, u, v, a[8], b[8];
    double len;
    int k;
    segment_frame(p0, p1, &d, &u, &v, &len);
    if (len < 1e-9)
        return;
    for (k = 0; k < 8; ++k) {
        a[k] = frame_point(p0, u, v, d, PX[k] * r0, PY[k] * r0 * aspect, 0.0);
        b[k] = frame_point(p0, u, v, d, PX[k] * r1, PY[k] * r1 * aspect, len);
    }
    glBegin(GL_QUADS);
    for (k = 0; k < 8; ++k) {
        int m = (k + 1) % 8;
        vec3 n = v3_normalize(v3_cross(v3_sub(a[m], a[k]), v3_sub(b[k], a[k])));
        /* point the normal away from the axis */
        if (v3_dot(n, v3_sub(a[k], p0)) < 0.0)
            n = v3_neg(n);
        vertex(a[k], n); vertex(a[m], n); vertex(b[m], n); vertex(b[k], n);
    }
    glEnd();
    glBegin(GL_POLYGON);
    for (k = 0; k < 8; ++k)
        vertex(a[k], v3_neg(d));
    glEnd();
    glBegin(GL_POLYGON);
    for (k = 0; k < 8; ++k)
        vertex(b[k], d);
    glEnd();
}

/* Each face is a grid whose vertices are pushed onto the rounded surface
 * (inner box + bevel along the normal). */
void arc_mesh_box(vec3 c, vec3 e)
{
    double ext[3], bevel = 0.22 * arc_min(e.x, arc_min(e.y, e.z)), s[3][6];
    int face, ax, i, j, k;
    ext[0] = e.x; ext[1] = e.y; ext[2] = e.z;
    for (k = 0; k < 3; ++k) {
        s[k][0] = -ext[k]; s[k][1] = -ext[k] + 0.3 * bevel; s[k][2] = -ext[k] + bevel;
        s[k][3] = ext[k] - bevel; s[k][4] = ext[k] - 0.3 * bevel; s[k][5] = ext[k];
    }
    for (face = 0; face < 6; ++face) {
        int a = face / 2, b = (a + 1) % 3, cc = (a + 2) % 3;
        double sign = face % 2 ? -1.0 : 1.0;
        glBegin(GL_QUADS);
        for (i = 0; i < 5; ++i) {
            for (j = 0; j < 5; ++j) {
                int corner;
                for (corner = 0; corner < 4; ++corner) {
                    int ii = i + (corner == 1 || corner == 2), jj = j + (corner >= 2);
                    double p[3], in[3], n[3];
                    vec3 nn;
                    p[a] = sign * ext[a];
                    p[b] = s[b][ii];
                    p[cc] = s[cc][jj];
                    for (ax = 0; ax < 3; ++ax) {
                        double lim = ext[ax] - bevel;
                        in[ax] = arc_clamp(p[ax], -lim, lim);
                        n[ax] = p[ax] - in[ax];
                    }
                    nn = v3_normalize(v3(n[0], n[1], n[2]));
                    vertex(v3(c.x + in[0] + bevel * nn.x, c.y + in[1] + bevel * nn.y,
                              c.z + in[2] + bevel * nn.z), nn);
                }
            }
        }
        glEnd();
    }
}

void arc_mesh_ring(vec3 c, vec3 axis, double major, double tube)
{
    vec3 d = v3_normalize(axis), u, v;
    int i, j, nu = 48, nv = 14;
    basis(d, &u, &v);
    for (i = 0; i < nu; ++i) {
        double t0 = 2.0 * ARC_PI * i / nu, t1 = 2.0 * ARC_PI * (i + 1) / nu;
        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= nv; ++j) {
            double p = 2.0 * ARC_PI * j / nv, cp = cos(p), sp = sin(p);
            vec3 e1 = v3_add(v3_scale(u, cos(t1)), v3_scale(v, sin(t1)));
            vec3 e0 = v3_add(v3_scale(u, cos(t0)), v3_scale(v, sin(t0)));
            vec3 n1 = v3_add(v3_scale(e1, cp), v3_scale(d, sp));
            vec3 n0 = v3_add(v3_scale(e0, cp), v3_scale(d, sp));
            vertex(v3_add(c, v3_add(v3_scale(e1, major), v3_scale(n1, tube))), n1);
            vertex(v3_add(c, v3_add(v3_scale(e0, major), v3_scale(n0, tube))), n0);
        }
        glEnd();
    }
}

/* ------------------------------------------------------------------------ */
/* Terrain */

static double cell_hash(int ix, int iy)
{
    unsigned long h = (unsigned long)(ix * 73856093L) ^ (unsigned long)(iy * 19349663L);
    h = (h ^ (h >> 13)) * 1274126177UL;
    return ((h >> 8) & 0xffff) / 65535.0;
}

static double occlusion(const arc_terrain *t, int ix, int iy, double h)
{
    return arc_clamp((arc_terrain_cell(t, ix, iy) - h) / 0.12, 0.0, 1.0);
}

/* Corner (sx, sy) of the top of column (ix, iy), at (x, y, h): color c,
 * darkened by ambient occlusion from the neighbours rising above it. */
static void top_corner(const arc_terrain *t, int ix, int iy, int sx, int sy, double x, double y,
                       double h, double c)
{
    double occ = occlusion(t, ix + sx, iy, h) + occlusion(t, ix, iy + sy, h)
                 + 0.5 * occlusion(t, ix + sx, iy + sy, h);
    double ao = 1.0 - 0.22 * occ;
    glColor3d(c * ao, c * ao, 1.04 * c * ao);
    glVertex3d(x, y, h);
}

static void side(double x0, double y0, double x1, double y1, double zlo, double zhi,
                 double nx, double ny, double c)
{
    glNormal3d(nx, ny, 0.0);
    glColor3d(0.45 * c, 0.45 * c, 0.47 * c); /* dark crevice at the foot */
    glVertex3d(x0, y0, zlo);
    glVertex3d(x1, y1, zlo);
    glColor3d(c, c, 1.04 * c);
    glVertex3d(x1, y1, zhi);
    glVertex3d(x0, y0, zhi);
}

void arc_mesh_terrain(const arc_terrain *t, double bottom)
{
    int ix, iy;
    glBegin(GL_QUADS);
    for (iy = 0; iy < t->ny; ++iy) {
        for (ix = 0; ix < t->nx; ++ix) {
            double h = arc_terrain_cell(t, ix, iy), n;
            double x0 = t->ox + ix * t->cell, x1 = x0 + t->cell;
            double y0 = t->oy + iy * t->cell, y1 = y0 + t->cell;
            double c = 0.36 + 0.05 * cell_hash(ix, iy) + 0.09 * arc_clamp(h, -0.3, 0.4);
            glNormal3d(0, 0, 1);
            top_corner(t, ix, iy, -1, -1, x0, y0, h, c);
            top_corner(t, ix, iy, 1, -1, x1, y0, h, c);
            top_corner(t, ix, iy, 1, 1, x1, y1, h, c);
            top_corner(t, ix, iy, -1, 1, x0, y1, h, c);
            c *= 0.8;
            if ((n = arc_terrain_cell(t, ix + 1, iy)) < h)
                side(x1, y0, x1, y1, arc_max(n, bottom), h, 1.0, 0.0, c);
            if ((n = arc_terrain_cell(t, ix - 1, iy)) < h)
                side(x0, y1, x0, y0, arc_max(n, bottom), h, -1.0, 0.0, c);
            if ((n = arc_terrain_cell(t, ix, iy + 1)) < h)
                side(x1, y1, x0, y1, arc_max(n, bottom), h, 0.0, 1.0, c);
            if ((n = arc_terrain_cell(t, ix, iy - 1)) < h)
                side(x0, y0, x1, y0, arc_max(n, bottom), h, 0.0, -1.0, c);
        }
    }
    glEnd();
}
