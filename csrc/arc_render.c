/*
 * arc_render.c - SDL2 + OpenGL 2.1 fixed function renderer.
 *
 * Scene: sky gradient, linear fog, one directional light, block terrain
 * compiled into a display list (rebuilt when the terrain changes), robot
 * visuals (spheres, boxes, cylinders, tapers, rings), blob shadows on the column tops under
 * every visual, and checkpoint pillars. Offscreen rendering draws into a
 * framebuffer object of a hidden window.
 */
#include <stdlib.h>
#include <string.h>
#include <SDL.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include "arc_render.h"

#define MAX_KEYS 32

typedef void (APIENTRYP gen_fn)(GLsizei, GLuint *);
typedef void (APIENTRYP bind_fn)(GLenum, GLuint);
typedef void (APIENTRYP storage_fn)(GLenum, GLenum, GLsizei, GLsizei);
typedef void (APIENTRYP attach_fn)(GLenum, GLenum, GLenum, GLuint);
typedef GLenum (APIENTRYP status_fn)(GLenum);
typedef void (APIENTRYP delete_fn)(GLsizei, const GLuint *);

struct arc_renderer {
    SDL_Window *win;
    SDL_GLContext ctx;
    int w, h, visible, closed;
    GLuint fbo, color_rb, depth_rb;
    GLuint list_sphere, list_cylinder, list_box, list_terrain;
    double terrain_key;
    /* camera: orbit around a smoothed target, behind the robot's heading */
    int cam_init, dragging, view;
    vec3 cam_target;
    double cam_heading, cam_yaw, cam_pitch, cam_dist;
    int keys[MAX_KEYS], n_keys;
    unsigned char *scratch;
    gen_fn gen_framebuffers, gen_renderbuffers;
    bind_fn bind_framebuffer, bind_renderbuffer;
    storage_fn renderbuffer_storage;
    attach_fn framebuffer_renderbuffer;
    status_fn check_framebuffer_status;
    delete_fn delete_framebuffers, delete_renderbuffers;
};

static const float SKY_TOP[3] = { 0.34f, 0.52f, 0.74f };
static const float SKY_HORIZON[3] = { 0.80f, 0.86f, 0.91f };

/* ------------------------------------------------------------------------ */
/* Meshes */

static void build_meshes(arc_renderer *r)
{
    int i, j, slices = 20, stacks = 12;

    r->list_sphere = glGenLists(4);
    r->list_cylinder = r->list_sphere + 1;
    r->list_box = r->list_sphere + 2;
    r->list_terrain = r->list_sphere + 3;

    /* unit sphere */
    glNewList(r->list_sphere, GL_COMPILE);
    for (i = 0; i < stacks; ++i) {
        double t0 = ARC_PI * i / stacks - 0.5 * ARC_PI;
        double t1 = ARC_PI * (i + 1) / stacks - 0.5 * ARC_PI;
        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= slices; ++j) {
            double p = 2.0 * ARC_PI * j / slices;
            double x0 = cos(t0) * cos(p), y0 = cos(t0) * sin(p), z0 = sin(t0);
            double x1 = cos(t1) * cos(p), y1 = cos(t1) * sin(p), z1 = sin(t1);
            glNormal3d(x1, y1, z1);
            glVertex3d(x1, y1, z1);
            glNormal3d(x0, y0, z0);
            glVertex3d(x0, y0, z0);
        }
        glEnd();
    }
    glEndList();

    /* unit cylinder along z, from z = 0 to 1, open ends */
    glNewList(r->list_cylinder, GL_COMPILE);
    glBegin(GL_QUAD_STRIP);
    for (j = 0; j <= slices; ++j) {
        double p = 2.0 * ARC_PI * j / slices;
        glNormal3d(cos(p), sin(p), 0.0);
        glVertex3d(cos(p), sin(p), 1.0);
        glVertex3d(cos(p), sin(p), 0.0);
    }
    glEnd();
    glEndList();

    /* unit cube [-1, 1]^3 */
    glNewList(r->list_box, GL_COMPILE);
    glBegin(GL_QUADS);
    glNormal3d(1, 0, 0);
    glVertex3d(1, -1, -1); glVertex3d(1, 1, -1); glVertex3d(1, 1, 1); glVertex3d(1, -1, 1);
    glNormal3d(-1, 0, 0);
    glVertex3d(-1, -1, -1); glVertex3d(-1, -1, 1); glVertex3d(-1, 1, 1); glVertex3d(-1, 1, -1);
    glNormal3d(0, 1, 0);
    glVertex3d(-1, 1, -1); glVertex3d(-1, 1, 1); glVertex3d(1, 1, 1); glVertex3d(1, 1, -1);
    glNormal3d(0, -1, 0);
    glVertex3d(-1, -1, -1); glVertex3d(1, -1, -1); glVertex3d(1, -1, 1); glVertex3d(-1, -1, 1);
    glNormal3d(0, 0, 1);
    glVertex3d(-1, -1, 1); glVertex3d(1, -1, 1); glVertex3d(1, 1, 1); glVertex3d(-1, 1, 1);
    glNormal3d(0, 0, -1);
    glVertex3d(-1, -1, -1); glVertex3d(-1, 1, -1); glVertex3d(1, 1, -1); glVertex3d(1, -1, -1);
    glEnd();
    glEndList();
}

/* ------------------------------------------------------------------------ */
/* Terrain */

static double cell_hash(int ix, int iy)
{
    unsigned long h = (unsigned long)(ix * 73856093L) ^ (unsigned long)(iy * 19349663L);
    h = (h ^ (h >> 13)) * 1274126177UL;
    return ((h >> 8) & 0xffff) / 65535.0;
}

static void terrain_side(double x0, double y0, double x1, double y1, double zlo, double zhi,
                         double nx, double ny)
{
    glNormal3d(nx, ny, 0.0);
    glVertex3d(x0, y0, zlo);
    glVertex3d(x1, y1, zlo);
    glVertex3d(x1, y1, zhi);
    glVertex3d(x0, y0, zhi);
}

/* Ground quad tessellated into tiles: fixed function fog is evaluated per
 * vertex, so a single huge quad would be fogged everywhere. */
static void ground_rect(double x0, double y0, double x1, double y1, double z)
{
    double tile = 3.0, x, y;
    for (y = y0; y < y1; y += tile) {
        for (x = x0; x < x1; x += tile) {
            double xe = arc_min(x + tile, x1), ye = arc_min(y + tile, y1);
            glVertex3d(x, y, z); glVertex3d(xe, y, z);
            glVertex3d(xe, ye, z); glVertex3d(x, ye, z);
        }
    }
}

static void build_terrain(arc_renderer *r, const arc_terrain *t)
{
    int ix, iy;
    double far = 90.0, z = t->outside;
    double gx0 = t->ox, gy0 = t->oy, gx1 = t->ox + t->nx * t->cell, gy1 = t->oy + t->ny * t->cell;

    glNewList(r->list_terrain, GL_COMPILE);
    glBegin(GL_QUADS);
    /* ground outside the grid, as four strips around it */
    glColor3f(0.80f, 0.78f, 0.74f);
    glNormal3d(0, 0, 1);
    ground_rect(-far, -far, gx0, far, z);
    ground_rect(gx1, -far, far, far, z);
    ground_rect(gx0, -far, gx1, gy0, z);
    ground_rect(gx0, gy1, gx1, far, z);

    /* columns, with a ring of outside columns for the boundary walls */
    for (iy = -1; iy <= t->ny; ++iy) {
        for (ix = -1; ix <= t->nx; ++ix) {
            double h = arc_terrain_cell(t, ix, iy), n;
            double x0 = t->ox + ix * t->cell, x1 = x0 + t->cell;
            double y0 = t->oy + iy * t->cell, y1 = y0 + t->cell;
            /* per-block variation plus a faint checkerboard, so motion stays
             * visible on flat ground */
            double c = 0.80 + 0.07 * cell_hash(ix, iy) + 0.06 * ((ix + iy) & 1)
                       + 0.25 * arc_clamp(h, -0.2, 0.4);
            int inside = ix >= 0 && iy >= 0 && ix < t->nx && iy < t->ny;
            c = arc_clamp(c, 0.6, 0.98);
            glColor3d(c, c * 0.975, c * 0.94);
            if (inside) {
                glNormal3d(0, 0, 1);
                glVertex3d(x0, y0, h); glVertex3d(x1, y0, h);
                glVertex3d(x1, y1, h); glVertex3d(x0, y1, h);
            }
            glColor3d(c * 0.92, c * 0.9, c * 0.86);
            if ((n = arc_terrain_cell(t, ix + 1, iy)) < h)
                terrain_side(x1, y0, x1, y1, n, h, 1.0, 0.0);
            if ((n = arc_terrain_cell(t, ix - 1, iy)) < h)
                terrain_side(x0, y1, x0, y0, n, h, -1.0, 0.0);
            if ((n = arc_terrain_cell(t, ix, iy + 1)) < h)
                terrain_side(x1, y1, x0, y1, n, h, 0.0, 1.0);
            if ((n = arc_terrain_cell(t, ix, iy - 1)) < h)
                terrain_side(x0, y0, x1, y0, n, h, 0.0, -1.0);
        }
    }
    glEnd();
    glEndList();
}

static double terrain_key(const arc_terrain *t)
{
    double k = t->nx * 1000.0 + t->ny + t->cell * 7.0 + t->outside * 13.0;
    int i;
    for (i = 0; i < t->nx * t->ny; ++i)
        k += t->h[i] * (1.0 + (i % 97) * 0.01);
    return k;
}

/* ------------------------------------------------------------------------ */
/* Drawing helpers */

static void mult_pose(vec3 x, quat q)
{
    GLdouble m[16];
    mat3 r;
    int i, k;
    m3_from_quat(&r, q);
    for (i = 0; i < 3; ++i)
        for (k = 0; k < 3; ++k)
            m[4 * k + i] = r.m[i][k]; /* column-major */
    m[3] = m[7] = m[11] = 0.0;
    m[12] = x.x;
    m[13] = x.y;
    m[14] = x.z;
    m[15] = 1.0;
    glMultMatrixd(m);
}

static void draw_sphere(const arc_renderer *r, vec3 c, double rad)
{
    glPushMatrix();
    glTranslated(c.x, c.y, c.z);
    glScaled(rad, rad, rad);
    glCallList(r->list_sphere);
    glPopMatrix();
}

/* Translates to p0 and rotates the z axis onto the unit vector d. */
static void align_z(vec3 p0, vec3 d)
{
    vec3 axis = v3_cross(v3(0, 0, 1), d);
    glTranslated(p0.x, p0.y, p0.z);
    if (v3_len(axis) > 1e-9)
        glRotated(acos(arc_clamp(d.z, -1.0, 1.0)) * 180.0 / ARC_PI, axis.x, axis.y, axis.z);
    else if (d.z < 0.0)
        glRotated(180.0, 1.0, 0.0, 0.0);
}

static void disc(double z, double rad, double nz)
{
    int j, slices = 24;
    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0, 0, nz);
    glVertex3d(0, 0, z);
    for (j = 0; j <= slices; ++j) {
        double p = 2.0 * ARC_PI * (nz > 0 ? j : slices - j) / slices;
        glVertex3d(rad * cos(p), rad * sin(p), z);
    }
    glEnd();
}

static void draw_cylinder(const arc_renderer *r, vec3 p0, vec3 p1, double rad)
{
    vec3 d = v3_sub(p1, p0);
    double len = v3_len(d);
    if (len < 1e-9)
        return;
    glPushMatrix();
    align_z(p0, v3_scale(d, 1.0 / len));
    disc(0.0, rad, -1.0);
    disc(len, rad, 1.0);
    glScaled(rad, rad, len);
    glCallList(r->list_cylinder);
    glPopMatrix();
}

/* Square prism along p0-p1 whose half width goes from r0 to r1. */
static void draw_taper(vec3 p0, vec3 p1, double r0, double r1)
{
    static const double SX[4] = { 1, -1, -1, 1 }, SY[4] = { 1, 1, -1, -1 };
    vec3 d = v3_sub(p1, p0);
    double len = v3_len(d);
    int k;
    if (len < 1e-9)
        return;
    glPushMatrix();
    align_z(p0, v3_scale(d, 1.0 / len));
    glBegin(GL_QUADS);
    for (k = 0; k < 4; ++k) {
        int m = (k + 1) % 4;
        /* outward normal of the face between corners k and m */
        vec3 a = v3(SX[k] * r0, SY[k] * r0, 0), b = v3(SX[m] * r0, SY[m] * r0, 0);
        vec3 c = v3(SX[m] * r1, SY[m] * r1, len);
        vec3 n = v3_normalize(v3_cross(v3_sub(b, a), v3_sub(c, a)));
        glNormal3d(n.x, n.y, n.z);
        glVertex3d(a.x, a.y, a.z);
        glVertex3d(b.x, b.y, b.z);
        glVertex3d(c.x, c.y, c.z);
        glVertex3d(SX[k] * r1, SY[k] * r1, len);
    }
    glNormal3d(0, 0, -1);
    for (k = 3; k >= 0; --k)
        glVertex3d(SX[k] * r0, SY[k] * r0, 0.0);
    glNormal3d(0, 0, 1);
    for (k = 0; k < 4; ++k)
        glVertex3d(SX[k] * r1, SY[k] * r1, len);
    glEnd();
    glPopMatrix();
}

/* Torus around the unit axis through c. */
static void draw_ring(vec3 c, vec3 axis, double major, double tube)
{
    int i, j, nu = 32, nv = 10;
    glPushMatrix();
    align_z(c, v3_normalize(axis));
    for (i = 0; i < nu; ++i) {
        double u0 = 2.0 * ARC_PI * i / nu, u1 = 2.0 * ARC_PI * (i + 1) / nu;
        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= nv; ++j) {
            double v = 2.0 * ARC_PI * j / nv, cv = cos(v), sv = sin(v);
            glNormal3d(cv * cos(u1), cv * sin(u1), sv);
            glVertex3d((major + tube * cv) * cos(u1), (major + tube * cv) * sin(u1), tube * sv);
            glNormal3d(cv * cos(u0), cv * sin(u0), sv);
            glVertex3d((major + tube * cv) * cos(u0), (major + tube * cv) * sin(u0), tube * sv);
        }
        glEnd();
    }
    glPopMatrix();
}

static void draw_visual(const arc_renderer *r, const arc_visual *v)
{
    if (v->emissive)
        glDisable(GL_LIGHTING);
    glColor3fv(v->color);
    switch (v->type) {
    case ARC_VIS_SPHERE:
        draw_sphere(r, v->p0, v->radius);
        break;
    case ARC_VIS_BOX:
        glPushMatrix();
        glTranslated(v->p0.x, v->p0.y, v->p0.z);
        glScaled(v->p1.x, v->p1.y, v->p1.z);
        glCallList(r->list_box);
        glPopMatrix();
        break;
    case ARC_VIS_TAPER:
        draw_taper(v->p0, v->p1, v->radius, v->radius2);
        break;
    case ARC_VIS_CYLINDER:
        draw_cylinder(r, v->p0, v->p1, v->radius);
        break;
    case ARC_VIS_RING:
        draw_ring(v->p0, v->p1, v->radius, v->radius2);
        break;
    }
    if (v->emissive)
        glEnable(GL_LIGHTING);
}

/* World-space center and footprint radius of a visual, for its shadow. */
static void visual_extent(const arc_world *w, const arc_visual *v, vec3 *c, double *rad)
{
    const arc_body *b = &w->bodies[v->body];
    vec3 local = v->p0;
    *rad = v->radius;
    if (v->type == ARC_VIS_TAPER || v->type == ARC_VIS_CYLINDER) {
        local = v3_scale(v3_add(v->p0, v->p1), 0.5);
        *rad = arc_max(v->radius, v->radius2) + 0.35 * v3_len(v3_sub(v->p1, v->p0));
    } else if (v->type == ARC_VIS_BOX) {
        *rad = 0.8 * arc_max(v->p1.x, v->p1.y);
    }
    *c = v3_add(b->x, q_rotate(b->q, local));
}

/* Soft dark disc of the given center alpha, fading out towards the rim. */
static void draw_disc(vec3 c, double rad, double alpha)
{
    int j, slices = 20;
    glBegin(GL_TRIANGLE_FAN);
    glColor4d(0.0, 0.0, 0.0, alpha);
    glVertex3d(c.x, c.y, c.z);
    glColor4d(0.0, 0.0, 0.0, 0.0);
    for (j = 0; j <= slices; ++j) {
        double p = 2.0 * ARC_PI * j / slices;
        glVertex3d(c.x + rad * cos(p), c.y + rad * sin(p), c.z);
    }
    glEnd();
}

/* Blob shadows on the column tops under every visual. */
static void draw_shadows(const arc_world *w)
{
    int i;
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.0f, -4.0f);
    for (i = 0; i < w->n_visuals; ++i) {
        vec3 c;
        double rad, ground, above;
        visual_extent(w, &w->visuals[i], &c, &rad);
        ground = arc_terrain_height(&w->terrain, c.x, c.y);
        above = c.z - ground;
        if (above < 0.0)
            continue;
        draw_disc(v3(c.x, c.y, ground + 0.002), 1.3 * rad + 0.35 * above,
                  0.45 * arc_clamp(1.0 - above / 1.6, 0.0, 1.0));
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

static void draw_markers(const arc_renderer *r, const double *m, int n, int active, double time)
{
    int i;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    for (i = 0; i < n; ++i) {
        vec3 base = v3(m[4 * i], m[4 * i + 1], m[4 * i + 2]);
        double radius = m[4 * i + 3];
        double bob = 0.06 * sin(3.0 * time);
        if (i == active)
            glColor4f(1.0f, 0.55f, 0.12f, 0.85f);
        else
            glColor4f(0.55f, 0.6f, 0.7f, 0.45f);
        draw_sphere(r, v3_add(base, v3(0, 0, 1.1 + bob)), i == active ? 0.14 : 0.1);
        glPushMatrix();
        glTranslated(base.x, base.y, base.z);
        glScaled(0.03, 0.03, 1.0);
        glCallList(r->list_cylinder);
        glPopMatrix();
        /* ground ring at the actual checkpoint radius */
        glPushMatrix();
        glTranslated(base.x, base.y, base.z + 0.01);
        glScaled(radius, radius, 0.02);
        glCallList(r->list_cylinder);
        glPopMatrix();
    }
    glDisable(GL_BLEND);
}

static void draw_sky(void)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glBegin(GL_QUADS);
    glColor3fv(SKY_HORIZON);
    glVertex2d(-1, -1); glVertex2d(1, -1);
    glColor3fv(SKY_TOP);
    glVertex2d(1, 1); glVertex2d(-1, 1);
    glEnd();
    glEnable(GL_FOG);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

static void look_at(vec3 eye, vec3 center, vec3 up)
{
    GLdouble m[16];
    vec3 f = v3_normalize(v3_sub(center, eye));
    vec3 s = v3_normalize(v3_cross(f, up));
    vec3 u = v3_cross(s, f);
    memset(m, 0, sizeof(m));
    m[0] = s.x; m[4] = s.y; m[8] = s.z;
    m[1] = u.x; m[5] = u.y; m[9] = u.z;
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z;
    m[15] = 1.0;
    glMultMatrixd(m);
    glTranslated(-eye.x, -eye.y, -eye.z);
}

/* ------------------------------------------------------------------------ */
/* Camera and input */

static double wrap_angle(double a)
{
    while (a > ARC_PI) a -= 2.0 * ARC_PI;
    while (a < -ARC_PI) a += 2.0 * ARC_PI;
    return a;
}

static void update_camera(arc_renderer *r, const arc_world *w)
{
    const arc_body *torso = &w->bodies[0];
    vec3 fwd = q_rotate(torso->q, v3(1, 0, 0));
    double heading = atan2(fwd.y, fwd.x);
    vec3 target = v3_add(torso->x, v3(0, 0, 0.1));
    if (!r->cam_init || v3_len(v3_sub(target, r->cam_target)) > 3.0) {
        r->cam_target = target;
        r->cam_heading = heading;
        r->cam_init = 1;
    }
    r->cam_target = v3_add(r->cam_target, v3_scale(v3_sub(target, r->cam_target), 0.15));
    r->cam_heading += 0.03 * wrap_angle(heading - r->cam_heading);
}

static void handle_events(arc_renderer *r)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_QUIT:
            r->closed = 1;
            break;
        case SDL_KEYDOWN:
            if (e.key.keysym.sym == SDLK_ESCAPE || e.key.keysym.sym == SDLK_q)
                r->closed = 1;
            if (e.key.keysym.sym == SDLK_c)
                arc_render_set_view(r, r->view + 1);
            if (r->n_keys < MAX_KEYS)
                r->keys[r->n_keys++] = (int)e.key.keysym.sym;
            break;
        case SDL_MOUSEBUTTONDOWN:
            if (e.button.button == SDL_BUTTON_LEFT)
                r->dragging = 1;
            break;
        case SDL_MOUSEBUTTONUP:
            if (e.button.button == SDL_BUTTON_LEFT)
                r->dragging = 0;
            break;
        case SDL_MOUSEMOTION:
            if (r->dragging) {
                r->cam_yaw -= 0.008 * e.motion.xrel;
                r->cam_pitch = arc_clamp(r->cam_pitch + 0.006 * e.motion.yrel, 0.02, 1.45);
            }
            break;
        case SDL_MOUSEWHEEL:
            r->cam_dist = arc_clamp(r->cam_dist * pow(0.9, e.wheel.y), 0.8, 30.0);
            break;
        case SDL_WINDOWEVENT:
            if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                SDL_GL_GetDrawableSize(r->win, &r->w, &r->h);
                free(r->scratch);
                r->scratch = 0;
            }
            break;
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Setup */

/* function pointers cannot be converted from void * in strict ISO C */
#define LOAD_GL(dst, name) \
    do { void *p_ = SDL_GL_GetProcAddress(name); memcpy(&(dst), &p_, sizeof(p_)); } while (0)

static int create_fbo(arc_renderer *r)
{
    LOAD_GL(r->gen_framebuffers, "glGenFramebuffers");
    LOAD_GL(r->gen_renderbuffers, "glGenRenderbuffers");
    LOAD_GL(r->bind_framebuffer, "glBindFramebuffer");
    LOAD_GL(r->bind_renderbuffer, "glBindRenderbuffer");
    LOAD_GL(r->renderbuffer_storage, "glRenderbufferStorage");
    LOAD_GL(r->framebuffer_renderbuffer, "glFramebufferRenderbuffer");
    LOAD_GL(r->check_framebuffer_status, "glCheckFramebufferStatus");
    LOAD_GL(r->delete_framebuffers, "glDeleteFramebuffers");
    LOAD_GL(r->delete_renderbuffers, "glDeleteRenderbuffers");
    if (!r->gen_framebuffers || !r->bind_framebuffer || !r->check_framebuffer_status)
        return 0;
    r->gen_framebuffers(1, &r->fbo);
    r->bind_framebuffer(GL_FRAMEBUFFER, r->fbo);
    r->gen_renderbuffers(1, &r->color_rb);
    r->bind_renderbuffer(GL_RENDERBUFFER, r->color_rb);
    r->renderbuffer_storage(GL_RENDERBUFFER, GL_RGBA8, r->w, r->h);
    r->framebuffer_renderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, r->color_rb);
    r->gen_renderbuffers(1, &r->depth_rb);
    r->bind_renderbuffer(GL_RENDERBUFFER, r->depth_rb);
    r->renderbuffer_storage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, r->w, r->h);
    r->framebuffer_renderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, r->depth_rb);
    return r->check_framebuffer_status(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
}

arc_renderer *arc_render_create(int width, int height, int visible)
{
    arc_renderer *r = (arc_renderer *)calloc(1, sizeof(arc_renderer));
    GLfloat ambient[4] = { 0.42f, 0.44f, 0.48f, 1.0f };
    GLfloat diffuse[4] = { 0.72f, 0.70f, 0.66f, 1.0f };
    GLfloat specular[4] = { 0.25f, 0.25f, 0.25f, 1.0f };
    if (!r)
        return 0;
    r->w = width;
    r->h = height;
    r->visible = visible;
    r->cam_dist = 4.0;
    arc_render_set_view(r, 0);

    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        free(r);
        return 0;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, visible ? 1 : 0);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, visible ? 4 : 0);
    r->win = SDL_CreateWindow("arc", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height,
                              SDL_WINDOW_OPENGL | (visible ? SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
                                                           : SDL_WINDOW_HIDDEN));
    if (!r->win) {
        /* retry without multisampling */
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);
        r->win = SDL_CreateWindow("arc", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width,
                                  height, SDL_WINDOW_OPENGL | (visible ? SDL_WINDOW_RESIZABLE : SDL_WINDOW_HIDDEN));
    }
    if (!r->win || !(r->ctx = SDL_GL_CreateContext(r->win))) {
        arc_render_destroy(r);
        return 0;
    }
    SDL_GL_MakeCurrent(r->win, r->ctx);
    if (visible) {
        SDL_GL_SetSwapInterval(1);
        SDL_GL_GetDrawableSize(r->win, &r->w, &r->h);
    } else if (!create_fbo(r)) {
        arc_render_destroy(r);
        return 0;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 24.0f);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogfv(GL_FOG_COLOR, SKY_HORIZON);
    glFogf(GL_FOG_START, 10.0f);
    glFogf(GL_FOG_END, 45.0f);
    glEnable(GL_FOG);
    if (visible)
        glEnable(GL_MULTISAMPLE);
    build_meshes(r);
    return r;
}

void arc_render_destroy(arc_renderer *r)
{
    if (!r)
        return;
    if (r->ctx) {
        SDL_GL_MakeCurrent(r->win, r->ctx);
        if (r->list_sphere)
            glDeleteLists(r->list_sphere, 4);
        if (r->fbo && r->delete_framebuffers) {
            r->delete_framebuffers(1, &r->fbo);
            r->delete_renderbuffers(1, &r->color_rb);
            r->delete_renderbuffers(1, &r->depth_rb);
        }
        SDL_GL_DeleteContext(r->ctx);
    }
    if (r->win)
        SDL_DestroyWindow(r->win);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    free(r->scratch);
    free(r);
}

void arc_render_size(const arc_renderer *r, int *width, int *height)
{
    *width = r->w;
    *height = r->h;
}

int arc_render_keys(arc_renderer *r, int *keys, int max)
{
    int n = r->n_keys < max ? r->n_keys : max;
    memcpy(keys, r->keys, sizeof(int) * (size_t)n);
    r->n_keys = 0;
    return n;
}

void arc_render_set_title(arc_renderer *r, const char *title)
{
    SDL_SetWindowTitle(r->win, title);
}

void arc_render_set_view(arc_renderer *r, int preset)
{
    static const double YAW[3] = { 0.5, 1.4, 2.6 };
    r->view = ((preset % 3) + 3) % 3;
    r->cam_yaw = YAW[r->view];
    r->cam_pitch = 0.3;
}

/* ------------------------------------------------------------------------ */
/* Frame */

int arc_render_frame(arc_renderer *r, const arc_sim *s, int world,
                     const double *markers, int n_markers, int active,
                     unsigned char *rgb)
{
    const arc_world *w = arc_sim_world(s, world);
    GLfloat light_dir[4] = { 0.45f, 0.3f, 0.85f, 0.0f };
    double key, yaw, near_plane = 0.05, aspect;
    vec3 eye;
    int i;

    handle_events(r);
    if (r->closed)
        return 0;
    SDL_GL_MakeCurrent(r->win, r->ctx);
    if (r->fbo)
        r->bind_framebuffer(GL_FRAMEBUFFER, r->fbo);

    key = terrain_key(&w->terrain);
    if (key != r->terrain_key) {
        build_terrain(r, &w->terrain);
        r->terrain_key = key;
    }
    update_camera(r, w);

    glViewport(0, 0, r->w, r->h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw_sky();

    aspect = (double)r->w / (r->h > 0 ? r->h : 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    {
        double top = near_plane * tan(0.5 * 50.0 * ARC_PI / 180.0);
        glFrustum(-top * aspect, top * aspect, -top, top, near_plane, 120.0);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    yaw = r->cam_heading + ARC_PI + r->cam_yaw;
    eye = v3_add(r->cam_target, v3(r->cam_dist * cos(r->cam_pitch) * cos(yaw),
                                   r->cam_dist * cos(r->cam_pitch) * sin(yaw),
                                   r->cam_dist * sin(r->cam_pitch)));
    /* keep the eye above the terrain */
    eye.z = arc_max(eye.z, arc_terrain_height(&w->terrain, eye.x, eye.y) + 0.2);
    look_at(eye, r->cam_target, v3(0, 0, 1));
    glLightfv(GL_LIGHT0, GL_POSITION, light_dir);

    glCallList(r->list_terrain);
    draw_shadows(w);

    for (i = 0; i < w->n_visuals; ++i) {
        const arc_visual *v = &w->visuals[i];
        const arc_body *b = &w->bodies[v->body];
        glPushMatrix();
        mult_pose(b->x, b->q);
        draw_visual(r, v);
        glPopMatrix();
    }
    if (markers && n_markers > 0)
        draw_markers(r, markers, n_markers, active, w->time);

    if (rgb) {
        int row, stride = r->w * 3;
        if (!r->scratch)
            r->scratch = (unsigned char *)malloc((size_t)stride * r->h);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(r->fbo ? GL_COLOR_ATTACHMENT0 : GL_BACK);
        glReadPixels(0, 0, r->w, r->h, GL_RGB, GL_UNSIGNED_BYTE, r->scratch);
        for (row = 0; row < r->h; ++row)
            memcpy(rgb + (size_t)row * stride, r->scratch + (size_t)(r->h - 1 - row) * stride, (size_t)stride);
    }
    if (r->visible)
        SDL_GL_SwapWindow(r->win);
    return 1;
}
