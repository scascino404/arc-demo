/*
 * arc_render.c - SDL2 + OpenGL 2.1 renderer with GLSL 1.20 shaders.
 *
 * Scene: an arena floating in pitch black void. One shadowed key
 * light (a 2048^2 shadow map following the camera), an unshadowed fill
 * light, hemisphere ambient, a grid of unseen lamps over the arena, a
 * spot light from the robot's eye, a point light from the checkpoint,
 * exponential fog into the dark beyond the camera target, and filmic tone
 * mapping, all per pixel in
 * one shader. Glowing parts get additive halos, and a vignette darkens the
 * frame corners.
 *
 * Geometry lives in display lists: the terrain (rebuilt when it changes),
 * one list per robot visual in its body frame (rebuilt when the visuals
 * change) and the checkpoint ring (rebuilt when its radius changes). The
 * view matrix is kept in GL_PROJECTION, so that gl_ModelViewMatrix maps
 * object to world space in the shaders. Uniforms that only depend on the
 * look or the terrain are set when the program or the terrain is built.
 * Offscreen rendering draws into a multisampled framebuffer object of a
 * hidden window, resolved into a single-sampled one when read back.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include "arc_render.h"

#define MAX_KEYS 32
#define SHADOW_SIZE 2048
#define MSAA 4                  /* samples per pixel, windows and offscreen */
#define SHADOW_RANGE 7.0        /* half width of the shadowed square around the camera target (m) */
#define VOID_DEPTH 6.0          /* terrain walls are drawn this far below the lowest column */
#define LAMP_GRID 3             /* LAMP_GRID^2 unseen lamps light the arena from above */
#define N_LAMPS (LAMP_GRID * LAMP_GRID) /* also hardcoded in SCENE_FS */
#define LAMP_HEIGHT 3.0         /* above the highest column under the lamp (m) */
#define BEACON_HEIGHT 1.1       /* checkpoint beacon above the ground (m) */

/* OpenGL 2.0+ entry points, loaded at runtime */
#define GL_FUNCS(X) \
    X(PFNGLCREATESHADERPROC, glCreateShader) \
    X(PFNGLSHADERSOURCEPROC, glShaderSource) \
    X(PFNGLCOMPILESHADERPROC, glCompileShader) \
    X(PFNGLGETSHADERIVPROC, glGetShaderiv) \
    X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog) \
    X(PFNGLDELETESHADERPROC, glDeleteShader) \
    X(PFNGLCREATEPROGRAMPROC, glCreateProgram) \
    X(PFNGLATTACHSHADERPROC, glAttachShader) \
    X(PFNGLLINKPROGRAMPROC, glLinkProgram) \
    X(PFNGLGETPROGRAMIVPROC, glGetProgramiv) \
    X(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog) \
    X(PFNGLDELETEPROGRAMPROC, glDeleteProgram) \
    X(PFNGLUSEPROGRAMPROC, glUseProgram) \
    X(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation) \
    X(PFNGLUNIFORM1IPROC, glUniform1i) \
    X(PFNGLUNIFORM1FPROC, glUniform1f) \
    X(PFNGLUNIFORM2FPROC, glUniform2f) \
    X(PFNGLUNIFORM3FPROC, glUniform3f) \
    X(PFNGLUNIFORM3FVPROC, glUniform3fv) \
    X(PFNGLUNIFORM4FPROC, glUniform4f) \
    X(PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv) \
    X(PFNGLACTIVETEXTUREPROC, glActiveTexture) \
    X(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers) \
    X(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer) \
    X(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers) \
    X(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D) \
    X(PFNGLFRAMEBUFFERRENDERBUFFERPROC, glFramebufferRenderbuffer) \
    X(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus) \
    X(PFNGLGENRENDERBUFFERSPROC, glGenRenderbuffers) \
    X(PFNGLBINDRENDERBUFFERPROC, glBindRenderbuffer) \
    X(PFNGLRENDERBUFFERSTORAGEPROC, glRenderbufferStorage) \
    X(PFNGLDELETERENDERBUFFERSPROC, glDeleteRenderbuffers) \
    X(PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC, glRenderbufferStorageMultisample) \
    X(PFNGLBLITFRAMEBUFFERPROC, glBlitFramebuffer)

#define GL_DECLARE(type, name) static type p_##name;
GL_FUNCS(GL_DECLARE)

typedef struct { double m[16]; } mat4; /* column-major, as OpenGL */

/* locations of the scene shader's uniforms that change between frames or
 * draws (the others are set once, see build_programs and update_terrain) */
typedef struct {
    GLint cam, fog, time, focus, shadow_mat, eye_pos, eye_dir, eye_col, lamp_pos, lamp_col;
    GLint mat, aux;
} scene_uniforms;

struct arc_renderer {
    SDL_Window *win;
    SDL_GLContext ctx;
    int w, h, visible, closed;
    GLuint fbo, color_rb, depth_rb;          /* offscreen target, multisampled */
    GLuint resolve_fbo, resolve_rb;          /* its resolved color, for readback */
    GLuint shadow_fbo, shadow_tex;
    GLuint prog_scene, prog_glow, prog_post;
    scene_uniforms su;
    GLint glow_cam, glow_fog, glow_beam, post_size;
    GLuint list_sphere, list_terrain, list_ring, list_visuals;
    arc_terrain terrain_cache;      /* what list_terrain shows */
    double ring_radius;             /* what list_ring shows */
    arc_visual vis_cache[ARC_MAX_VISUALS];
    int n_vis_cache;
    /* camera: orbit around a smoothed target, behind the robot's heading */
    int cam_init, dragging, view;
    vec3 cam_target;
    double cam_heading, cam_yaw, cam_pitch, cam_dist;
    int keys[MAX_KEYS], n_keys;
    unsigned char *scratch;
};

/* ------------------------------------------------------------------------ */
/* Look. Colors are linear; the shaders tone map and gamma correct. */

static const float SUN_DIR[3] = { 0.48f, 0.30f, 0.82f };     /* towards the light */
static const float SUN_COL[3] = { 1.45f, 1.55f, 1.80f };
static const float FILL_DIR[3] = { -0.48f, -0.30f, 0.45f };   /* opposite the key light */
static const float FILL_COL[3] = { 0.16f, 0.13f, 0.11f };
static const float LAMP_COL[3] = { 0.40f, 0.30f, 0.20f };     /* unseen lamps, warm */
static const float SKY_COL[3] = { 0.090f, 0.100f, 0.125f };   /* ambient from above */
static const float GND_COL[3] = { 0.030f, 0.030f, 0.034f };   /* ambient from below */
static const float FOG_DENSITY = 0.05f;   /* into black, per meter beyond the camera target's distance */
static const float EYE_LIGHT = 2.2f;                          /* eye spot intensity */
static const float LAMP_ACTIVE[3] = { 1.00f, 0.42f, 0.08f };  /* checkpoint, seen */
static const float LAMP_IDLE[3] = { 0.20f, 0.55f, 1.00f };    /* checkpoint, unseen */

/* per material: specular, glossiness, emission, pattern (1 terrain, 2 panels) */
static const float MATERIALS[][4] = {
    { 0.45f, 36.0f, 0.0f, 0.0f },   /* ARC_MAT_PAINT */
    { 0.50f, 40.0f, 0.0f, 2.0f },   /* ARC_MAT_PANELS */
    { 0.90f, 64.0f, 0.0f, 0.0f },   /* ARC_MAT_METAL */
    { 0.08f, 10.0f, 0.0f, 0.0f },   /* ARC_MAT_RUBBER */
    { 1.80f, 180.0f, 0.0f, 0.0f },  /* ARC_MAT_LENS */
    { 0.60f, 40.0f, 0.0f, 0.0f },   /* ARC_MAT_RIBS */
    { 0.00f, 1.0f, 3.0f, 0.0f }     /* ARC_MAT_GLOW */
};

/* ------------------------------------------------------------------------ */
/* Shaders (GLSL 1.20). Sources are arrays of lines: ISO C90 limits string
 * literals to 509 characters. */

static const char *SCENE_VS[] = {
    "#version 120\n",
    "varying vec3 v_pos, v_nrm, v_obj;\n",
    "varying vec4 v_col;\n",
    "void main() {\n",
    "    vec4 p = gl_ModelViewMatrix * gl_Vertex;\n", /* world space */
    "    v_pos = p.xyz;\n",
    "    v_nrm = gl_NormalMatrix * gl_Normal;\n",
    "    v_obj = gl_Vertex.xyz;\n",
    "    v_col = gl_Color;\n",
    "    gl_Position = gl_ProjectionMatrix * p;\n",
    "}\n"
};

static const char *SCENE_FS[] = {
    "#version 120\n",
    "uniform vec3 u_cam, u_sun_dir, u_sun_col, u_sky_col, u_gnd_col;\n",
    "uniform vec2 u_fog;\n",          /* density, distance where it starts */
    "uniform float u_floor_z, u_time;\n",
    "uniform vec2 u_focus;\n",        /* the key light falls off away from here */
    "uniform sampler2DShadow u_shadow;\n",
    "uniform mat4 u_shadow_mat;\n",
    "uniform vec2 u_shadow_bias;\n",   /* texel size, normal offset (m) */
    "uniform vec3 u_eye_pos, u_eye_dir, u_eye_col, u_lamp_pos, u_lamp_col;\n",
    "uniform vec3 u_fill_dir, u_fill_col, u_lamps_col;\n",
    "uniform vec3 u_lamps_pos[9];\n",  /* N_LAMPS */
    "uniform vec4 u_mat;\n",           /* specular, gloss, emission, pattern */
    "uniform vec4 u_aux;\n",           /* terrain: grid origin, cell; else rim */
    "varying vec3 v_pos, v_nrm, v_obj;\n",
    "varying vec4 v_col;\n",
    "float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n",
    "float noise(vec2 p) {\n",
    "    vec2 i = floor(p), f = fract(p);\n",
    "    f = f * f * (3.0 - 2.0 * f);\n",
    "    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),\n",
    "               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);\n",
    "}\n",
    "float lit(vec3 p, vec3 n) {\n",
    "    vec4 s = u_shadow_mat * vec4(p + n * u_shadow_bias.y, 1.0);\n",
    "    float t = u_shadow_bias.x, k = 0.0;\n",
    "    k += shadow2D(u_shadow, s.xyz + vec3(-0.6, -0.9, 0.0) * t).r;\n",
    "    k += shadow2D(u_shadow, s.xyz + vec3(0.9, -0.6, 0.0) * t).r;\n",
    "    k += shadow2D(u_shadow, s.xyz + vec3(-0.9, 0.6, 0.0) * t).r;\n",
    "    k += shadow2D(u_shadow, s.xyz + vec3(0.6, 0.9, 0.0) * t).r;\n",
    "    vec2 e = abs(s.xy - 0.5);\n",  /* fade out at the edge of the map */
    "    return mix(1.0, 0.25 * k, smoothstep(0.5, 0.42, max(e.x, e.y)));\n",
    "}\n",
    /* Blinn-Phong diffuse + specular of one light */
    "vec3 shade(vec3 a, float spec, float gloss, vec3 n, vec3 v, vec3 l, vec3 c) {\n",
    "    float d = max(dot(n, l), 0.0);\n",
    "    float s = pow(max(dot(n, normalize(l + v)), 0.0), gloss) * (gloss + 8.0) / 25.0;\n",
    "    return (a + spec * s) * d * c;\n",
    "}\n",
    "void main() {\n",
    "    vec3 n = normalize(v_nrm), v = normalize(u_cam - v_pos);\n",
    "    vec3 a = v_col.rgb * v_col.rgb;\n", /* albedo, roughly linear */
    "    float spec = u_mat.x, gloss = u_mat.y, pat = u_mat.w, rim = u_aux.w;\n",
    "    if (pat > 0.5 && pat < 1.5) {\n",  /* terrain: grime and block seams */
    "        vec2 g = (v_pos.xy - u_aux.xy) / u_aux.z;\n",
    "        vec2 f = 0.5 - abs(fract(g) - 0.5);\n",
    "        float edge = min(f.x, f.y), w = fwidth(g.x) + fwidth(g.y);\n",
    "        float grime = 0.6 * noise(v_pos.xy * 1.7) + 0.4 * noise(v_pos.xy * 6.0 + v_pos.z * 4.0);\n",
    "        a *= mix(0.65, 1.25, grime);\n",
    "        if (n.z > 0.7) a *= mix(0.35, 1.0, smoothstep(0.0, 0.02 + w, edge));\n",
    "        spec *= grime;\n",
    "        rim = 0.0;\n",
    "    } else if (pat > 1.5) {\n",         /* unit sphere: staggered panel seams */
    "        vec3 o = normalize(v_obj);\n",
    "        float lat = asin(clamp(o.z, -1.0, 1.0)) * 7.0 / 3.14159;\n",
    "        float lon = atan(o.y, o.x) * 6.0 / 3.14159 + 0.5 * floor(lat);\n",
    "        float wl = fwidth(lat) * 1.2, wn = fwidth(lon) * 1.2;\n",
    "        float s1 = 1.0 - smoothstep(0.0, 0.04 + wl, 0.5 - abs(fract(lat) - 0.5));\n",
    "        float s2 = (1.0 - smoothstep(0.0, 0.03 + wn, 0.5 - abs(fract(lon) - 0.5)))\n",
    "                   * step(abs(o.z), 0.92);\n",
    "        a *= 1.0 - 0.7 * max(s1, s2);\n",
    "        spec *= 1.0 - 0.8 * max(s1, s2);\n",
    "    }\n",
    "    if (rim > 0.5) {\n",              /* robot: worn paint */
    "        float wear = 0.6 * noise(v_obj.xy * 30.0 + v_obj.z * 17.0)\n",
    "                     + 0.4 * noise(v_obj.yz * 90.0 + v_obj.x * 41.0);\n",
    "        a *= mix(0.7, 1.15, wear);\n",
    "        spec *= mix(0.4, 1.4, wear);\n",
    "    }\n",
    "    vec3 c = a * mix(u_gnd_col, u_sky_col, 0.5 + 0.5 * n.z);\n",
    "    vec2 q = (v_pos.xy - u_focus) / 6.0;\n",
    "    float pool = 0.65 + 0.35 * exp(-dot(q, q));\n",
    "    c += shade(a, spec, gloss, n, v, u_sun_dir, u_sun_col) * lit(v_pos, n) * pool;\n",
    "    c += shade(a, spec, gloss, n, v, u_fill_dir, u_fill_col);\n",
    "    vec3 l;\n",
    "    float d2;\n",
    "    for (int i = 0; i < 9; ++i) {\n",  /* lamps above, shining down */
    "        l = u_lamps_pos[i] - v_pos;\n",
    "        d2 = dot(l, l);\n",
    "        if (d2 > 64.0) continue;\n",  /* out of reach: below 0.25 of the cone edge */
    "        l *= inversesqrt(d2);\n",
    "        c += shade(a, spec, gloss, n, v, l, u_lamps_col) * smoothstep(0.25, 0.8, l.z) / (1.0 + 0.12 * d2);\n",
    "    }\n",
    "    l = u_eye_pos - v_pos;\n",   /* eye: a wide spot light */
    "    d2 = dot(l, l);\n",
    "    l *= inversesqrt(d2);\n",
    "    float cone = smoothstep(0.35, 0.85, dot(-l, u_eye_dir));\n",
    "    c += shade(a, spec, gloss, n, v, l, u_eye_col) * cone / (1.0 + 2.0 * d2);\n",
    "    l = u_lamp_pos - v_pos;\n",       /* checkpoint lamp */
    "    d2 = dot(l, l);\n",
    "    c += shade(a, spec, gloss, n, v, l * inversesqrt(d2), u_lamp_col) / (1.0 + 1.5 * d2);\n",
    "    float nv = 1.0 - max(dot(n, v), 0.0);\n",
    "    vec3 r = reflect(-v, n);\n",       /* sheen of the surroundings */
    "    c += spec * (0.04 + 0.5 * pow(nv, 5.0)) * mix(u_gnd_col, 3.0 * u_sky_col, 0.5 + 0.5 * r.z);\n",
    "    c += rim * pow(nv, 4.0) * 0.6 * u_sky_col;\n",
    "    c = mix(c, a * u_mat.z, step(0.5, u_mat.z));\n", /* emissive */
    "    c *= exp(-u_fog.x * max(length(u_cam - v_pos) - u_fog.y, 0.0));\n", /* fog */
    "    c *= smoothstep(u_floor_z - 3.0, u_floor_z + 0.2, v_pos.z);\n", /* into the void */
    "    c = c * (2.51 * c + 0.03) / (c * (2.43 * c + 0.59) + 0.14);\n", /* ACES fit */
    "    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));\n",
    "    c += (hash(gl_FragCoord.xy + fract(u_time) * 61.0) - 0.5) / 255.0;\n", /* dither */
    "    gl_FragColor = vec4(c, 1.0);\n",
    "}\n"
};

/* additive glow: soft round halos, or vertical beams (u_beam = 1) */
static const char *GLOW_VS[] = {
    "#version 120\n",
    "varying vec2 v_uv;\n",
    "varying vec4 v_col;\n",
    "varying vec3 v_pos;\n",
    "void main() {\n",
    "    v_uv = gl_MultiTexCoord0.xy;\n",
    "    v_col = gl_Color;\n",
    "    v_pos = gl_Vertex.xyz;\n",
    "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n",
    "}\n"
};

static const char *GLOW_FS[] = {
    "#version 120\n",
    "uniform vec3 u_cam;\n",
    "uniform vec2 u_fog;\n",
    "uniform float u_beam;\n",
    "varying vec2 v_uv;\n",
    "varying vec4 v_col;\n",
    "varying vec3 v_pos;\n",
    "void main() {\n",
    "    float r2 = dot(v_uv, v_uv);\n",
    "    float halo = exp(-6.0 * r2) * 0.8 + exp(-30.0 * r2);\n",
    "    float beam = exp(-12.0 * v_uv.x * v_uv.x) * (1.0 - v_uv.y) * (1.0 - v_uv.y);\n",
    "    float k = mix(halo * (1.0 - smoothstep(0.7, 1.0, r2)), beam, u_beam);\n",
    "    k *= v_col.a * exp(-u_fog.x * max(length(u_cam - v_pos) - u_fog.y, 0.0));\n",
    "    gl_FragColor = vec4(v_col.rgb * k, 1.0);\n",
    "}\n"
};

/* vignette, multiplied onto the frame */
static const char *POST_VS[] = {
    "#version 120\n",
    "void main() { gl_Position = gl_Vertex; }\n"
};

static const char *POST_FS[] = {
    "#version 120\n",
    "uniform vec2 u_size;\n",
    "void main() {\n",
    "    vec2 p = gl_FragCoord.xy / u_size - 0.5;\n",
    "    p.x *= u_size.x / u_size.y;\n",
    "    gl_FragColor = vec4(vec3(1.0 - 0.4 * smoothstep(0.45, 1.1, length(p))), 1.0);\n",
    "}\n"
};

#define COUNT(a) ((GLsizei)(sizeof(a) / sizeof((a)[0])))

static GLuint compile(GLenum type, const char **src, GLsizei n)
{
    GLuint s = p_glCreateShader(type);
    GLint ok = 0;
    p_glShaderSource(s, n, src, 0);
    p_glCompileShader(s);
    p_glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        p_glGetShaderInfoLog(s, sizeof(log), 0, log);
        fprintf(stderr, "arc_render: shader compilation failed:\n%s\n", log);
        p_glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint program(const char **vs, GLsizei nv, const char **fs, GLsizei nf)
{
    GLuint v = compile(GL_VERTEX_SHADER, vs, nv), f = compile(GL_FRAGMENT_SHADER, fs, nf), p;
    GLint ok = 0;
    if (!v || !f)
        return 0;
    p = p_glCreateProgram();
    p_glAttachShader(p, v);
    p_glAttachShader(p, f);
    p_glLinkProgram(p);
    p_glDeleteShader(v);
    p_glDeleteShader(f);
    p_glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        p_glGetProgramInfoLog(p, sizeof(log), 0, log);
        fprintf(stderr, "arc_render: shader linking failed:\n%s\n", log);
        p_glDeleteProgram(p);
        return 0;
    }
    return p;
}

static void uniform_dir(GLuint p, const char *name, const float *d)
{
    vec3 n = v3_normalize(v3(d[0], d[1], d[2]));
    p_glUniform3f(p_glGetUniformLocation(p, name), (float)n.x, (float)n.y, (float)n.z);
}

static int build_programs(arc_renderer *r)
{
    scene_uniforms *u = &r->su;
    GLuint p;
    r->prog_scene = program(SCENE_VS, COUNT(SCENE_VS), SCENE_FS, COUNT(SCENE_FS));
    r->prog_glow = program(GLOW_VS, COUNT(GLOW_VS), GLOW_FS, COUNT(GLOW_FS));
    r->prog_post = program(POST_VS, COUNT(POST_VS), POST_FS, COUNT(POST_FS));
    if (!r->prog_scene || !r->prog_glow || !r->prog_post)
        return 0;
    p = r->prog_scene;
#define LOC(field, name) u->field = p_glGetUniformLocation(p, name)
    LOC(cam, "u_cam"); LOC(fog, "u_fog"); LOC(time, "u_time"); LOC(focus, "u_focus");
    LOC(shadow_mat, "u_shadow_mat"); LOC(eye_pos, "u_eye_pos"); LOC(eye_dir, "u_eye_dir");
    LOC(eye_col, "u_eye_col"); LOC(lamp_pos, "u_lamp_pos"); LOC(lamp_col, "u_lamp_col");
    LOC(mat, "u_mat"); LOC(aux, "u_aux");
#undef LOC
    /* the look: constant */
    p_glUseProgram(p);
    uniform_dir(p, "u_sun_dir", SUN_DIR);
    uniform_dir(p, "u_fill_dir", FILL_DIR);
    p_glUniform3fv(p_glGetUniformLocation(p, "u_sun_col"), 1, SUN_COL);
    p_glUniform3fv(p_glGetUniformLocation(p, "u_fill_col"), 1, FILL_COL);
    p_glUniform3fv(p_glGetUniformLocation(p, "u_lamps_col"), 1, LAMP_COL);
    p_glUniform3fv(p_glGetUniformLocation(p, "u_sky_col"), 1, SKY_COL);
    p_glUniform3fv(p_glGetUniformLocation(p, "u_gnd_col"), 1, GND_COL);
    p_glUniform1i(p_glGetUniformLocation(p, "u_shadow"), 0);
    p_glUniform2f(p_glGetUniformLocation(p, "u_shadow_bias"), 1.0f / SHADOW_SIZE,
                  (float)(3.0 * SHADOW_RANGE / SHADOW_SIZE));
    p_glUseProgram(0);
    r->glow_cam = p_glGetUniformLocation(r->prog_glow, "u_cam");
    r->glow_fog = p_glGetUniformLocation(r->prog_glow, "u_fog");
    r->glow_beam = p_glGetUniformLocation(r->prog_glow, "u_beam");
    r->post_size = p_glGetUniformLocation(r->prog_post, "u_size");
    return 1;
}

/* ------------------------------------------------------------------------ */
/* Matrices */

static mat4 m4_mul(mat4 a, mat4 b)
{
    mat4 r;
    int i, j, k;
    for (j = 0; j < 4; ++j)
        for (i = 0; i < 4; ++i) {
            double s = 0.0;
            for (k = 0; k < 4; ++k)
                s += a.m[4 * k + i] * b.m[4 * j + k];
            r.m[4 * j + i] = s;
        }
    return r;
}

static mat4 m4_look_at(vec3 eye, vec3 center, vec3 up)
{
    mat4 m;
    vec3 f = v3_normalize(v3_sub(center, eye));
    vec3 s = v3_normalize(v3_cross(f, up));
    vec3 u = v3_cross(s, f);
    memset(&m, 0, sizeof(m));
    m.m[0] = s.x; m.m[4] = s.y; m.m[8] = s.z;
    m.m[1] = u.x; m.m[5] = u.y; m.m[9] = u.z;
    m.m[2] = -f.x; m.m[6] = -f.y; m.m[10] = -f.z;
    m.m[12] = -v3_dot(s, eye);
    m.m[13] = -v3_dot(u, eye);
    m.m[14] = v3_dot(f, eye);
    m.m[15] = 1.0;
    return m;
}

static mat4 m4_frustum(double l, double r, double b, double t, double n, double f)
{
    mat4 m;
    memset(&m, 0, sizeof(m));
    m.m[0] = 2.0 * n / (r - l);
    m.m[5] = 2.0 * n / (t - b);
    m.m[8] = (r + l) / (r - l);
    m.m[9] = (t + b) / (t - b);
    m.m[10] = -(f + n) / (f - n);
    m.m[11] = -1.0;
    m.m[14] = -2.0 * f * n / (f - n);
    return m;
}

static mat4 m4_ortho(double l, double r, double b, double t, double n, double f)
{
    mat4 m;
    memset(&m, 0, sizeof(m));
    m.m[0] = 2.0 / (r - l);
    m.m[5] = 2.0 / (t - b);
    m.m[10] = -2.0 / (f - n);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = -(t + b) / (t - b);
    m.m[14] = -(f + n) / (f - n);
    m.m[15] = 1.0;
    return m;
}

static vec3 m4_point(mat4 a, vec3 p)
{
    return v3(a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z + a.m[12],
              a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z + a.m[13],
              a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z + a.m[14]);
}

static void mult_pose(vec3 x, quat q)
{
    GLdouble m[16];
    mat3 r;
    int i, k;
    m3_from_quat(&r, q);
    for (i = 0; i < 3; ++i)
        for (k = 0; k < 3; ++k)
            m[4 * k + i] = r.m[i][k];
    m[3] = m[7] = m[11] = 0.0;
    m[12] = x.x;
    m[13] = x.y;
    m[14] = x.z;
    m[15] = 1.0;
    glMultMatrixd(m);
}

/* ------------------------------------------------------------------------ */
/* Meshes. Visuals are emitted in their body frame; the terrain in world
 * space. Face culling is off, so winding does not matter. */

static void vertex(vec3 p, vec3 n)
{
    glNormal3d(n.x, n.y, n.z);
    glVertex3d(p.x, p.y, p.z);
}

static void unit_sphere(void)
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

/* Closed cylinder; ribbed ones get a corrugated surface. */
static void mesh_cylinder(vec3 p0, vec3 p1, double rad, int ribbed)
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

/* Faceted prism: a rectangle of half extents (w, aspect w) with chamfered
 * corners, w going from r0 at p0 to r1 at p1. */
static void mesh_taper(vec3 p0, vec3 p1, double r0, double r1, double aspect)
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

/* Box with rounded edges: each face is a grid whose vertices are pushed
 * onto the rounded surface (inner box + bevel along the normal). */
static void mesh_box(vec3 c, vec3 e)
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

static void mesh_ring(vec3 c, vec3 axis, double major, double tube)
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

static void build_visual(const arc_renderer *r, const arc_visual *v)
{
    switch (v->type) {
    case ARC_VIS_SPHERE:
        glPushMatrix();
        glTranslated(v->p0.x, v->p0.y, v->p0.z);
        glScaled(v->radius, v->radius, v->radius);
        glCallList(r->list_sphere);
        glPopMatrix();
        break;
    case ARC_VIS_BOX:
        mesh_box(v->p0, v->p1);
        break;
    case ARC_VIS_TAPER:
        mesh_taper(v->p0, v->p1, v->radius, v->radius2, v->aspect);
        break;
    case ARC_VIS_CYLINDER:
        mesh_cylinder(v->p0, v->p1, v->radius, v->material == ARC_MAT_RIBS);
        break;
    case ARC_VIS_RING:
        mesh_ring(v->p0, v->p1, v->radius, v->radius2);
        break;
    }
}

/* Recompiles the visual lists when the robot's visuals change. */
static void update_visuals(arc_renderer *r, const arc_world *w)
{
    int i, n = w->n_visuals;
    if (n == r->n_vis_cache && !memcmp(r->vis_cache, w->visuals, sizeof(arc_visual) * (size_t)n))
        return;
    for (i = 0; i < n; ++i) {
        glNewList(r->list_visuals + i, GL_COMPILE);
        build_visual(r, &w->visuals[i]);
        glEndList();
    }
    memcpy(r->vis_cache, w->visuals, sizeof(arc_visual) * (size_t)n);
    r->n_vis_cache = n;
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

/* Ambient occlusion at the corner (sx, sy) of a column top, from the
 * neighbours rising above it. */
static double corner_ao(const arc_terrain *t, int ix, int iy, int sx, int sy, double h)
{
    double occ = occlusion(t, ix + sx, iy, h) + occlusion(t, ix, iy + sy, h)
                 + 0.5 * occlusion(t, ix + sx, iy + sy, h);
    return 1.0 - 0.22 * occ;
}

static void terrain_side(double x0, double y0, double x1, double y1, double zlo, double zhi,
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

/* Recompiles the terrain list, and sets the scene shader's uniforms that
 * depend on the terrain, when the terrain changes. */
static void update_terrain(arc_renderer *r, const arc_terrain *t)
{
    int ix, iy, k;
    double lo = 1e30, bottom;
    float lamps[3 * N_LAMPS];
    if (!memcmp(&r->terrain_cache, t, sizeof(*t)))
        return;
    r->terrain_cache = *t;
    for (k = 0; k < t->nx * t->ny; ++k)
        lo = arc_min(lo, t->h[k]);
    bottom = lo - VOID_DEPTH;
    for (k = 0; k < N_LAMPS; ++k) {
        double x = t->ox + t->nx * t->cell * (k % LAMP_GRID + 0.5) / LAMP_GRID;
        double y = t->oy + t->ny * t->cell * (k / LAMP_GRID + 0.5) / LAMP_GRID, top = lo, dx, dy;
        for (dx = -1.0; dx <= 1.0; dx += 0.5 * t->cell)
            for (dy = -1.0; dy <= 1.0; dy += 0.5 * t->cell)
                top = arc_max(top, arc_terrain_height(t, x + dx, y + dy));
        lamps[3 * k] = (float)x;
        lamps[3 * k + 1] = (float)y;
        lamps[3 * k + 2] = (float)(top + LAMP_HEIGHT);
    }
    p_glUseProgram(r->prog_scene);
    p_glUniform1f(p_glGetUniformLocation(r->prog_scene, "u_floor_z"), (float)lo);
    p_glUniform3fv(p_glGetUniformLocation(r->prog_scene, "u_lamps_pos"), N_LAMPS, lamps);
    p_glUseProgram(0);

    glNewList(r->list_terrain, GL_COMPILE);
    glBegin(GL_QUADS);
    for (iy = 0; iy < t->ny; ++iy) {
        for (ix = 0; ix < t->nx; ++ix) {
            double h = arc_terrain_cell(t, ix, iy), n;
            double x0 = t->ox + ix * t->cell, x1 = x0 + t->cell;
            double y0 = t->oy + iy * t->cell, y1 = y0 + t->cell;
            double c = 0.36 + 0.05 * cell_hash(ix, iy) + 0.09 * arc_clamp(h, -0.3, 0.4), ao;
            glNormal3d(0, 0, 1);
            ao = corner_ao(t, ix, iy, -1, -1, h);
            glColor3d(c * ao, c * ao, 1.04 * c * ao);
            glVertex3d(x0, y0, h);
            ao = corner_ao(t, ix, iy, 1, -1, h);
            glColor3d(c * ao, c * ao, 1.04 * c * ao);
            glVertex3d(x1, y0, h);
            ao = corner_ao(t, ix, iy, 1, 1, h);
            glColor3d(c * ao, c * ao, 1.04 * c * ao);
            glVertex3d(x1, y1, h);
            ao = corner_ao(t, ix, iy, -1, 1, h);
            glColor3d(c * ao, c * ao, 1.04 * c * ao);
            glVertex3d(x0, y1, h);
            c *= 0.8;
            if ((n = arc_terrain_cell(t, ix + 1, iy)) < h)
                terrain_side(x1, y0, x1, y1, arc_max(n, bottom), h, 1.0, 0.0, c);
            if ((n = arc_terrain_cell(t, ix - 1, iy)) < h)
                terrain_side(x0, y1, x0, y0, arc_max(n, bottom), h, -1.0, 0.0, c);
            if ((n = arc_terrain_cell(t, ix, iy + 1)) < h)
                terrain_side(x1, y1, x0, y1, arc_max(n, bottom), h, 0.0, 1.0, c);
            if ((n = arc_terrain_cell(t, ix, iy - 1)) < h)
                terrain_side(x0, y0, x1, y0, arc_max(n, bottom), h, 0.0, -1.0, c);
        }
    }
    glEnd();
    glEndList();
}

/* ------------------------------------------------------------------------ */
/* Drawing */

static vec3 visual_center(const arc_world *w, const arc_visual *v)
{
    const arc_body *b = &w->bodies[v->body];
    vec3 local = v->type == ARC_VIS_CYLINDER || v->type == ARC_VIS_TAPER
                 ? v3_scale(v3_add(v->p0, v->p1), 0.5) : v->p0;
    return v3_add(b->x, q_rotate(b->q, local));
}

static void set_material(const arc_renderer *r, int material, const float *color, double rim)
{
    const float *m = MATERIALS[material >= 0 && material <= ARC_MAT_GLOW ? material : 0];
    glColor3fv(color);
    p_glUniform4f(r->su.mat, m[0], m[1], m[2], m[3]);
    p_glUniform4f(r->su.aux, 0.0f, 0.0f, 1.0f, (float)rim);
}

/* lit = 0: geometry only, without the glowing parts (for the shadow map) */
static void draw_robot(const arc_renderer *r, const arc_world *w, int lit)
{
    int i;
    for (i = 0; i < w->n_visuals; ++i) {
        const arc_visual *v = &w->visuals[i];
        const arc_body *b = &w->bodies[v->body];
        if (!lit && v->material == ARC_MAT_GLOW)
            continue;
        if (lit)
            set_material(r, v->material, v->color, 1.0);
        glPushMatrix();
        mult_pose(b->x, b->q);
        glCallList(r->list_visuals + i);
        glPopMatrix();
    }
}

/* Checkpoint beacon of marker m, bobbing above the ground. */
static vec3 beacon(const double *m, double time)
{
    return v3(m[0], m[1], m[2] + BEACON_HEIGHT + 0.06 * sin(3.0 * time));
}

/* Checkpoints: a glowing beacon and a ring on the ground. */
static void draw_markers(arc_renderer *r, const double *m, int n, int active, double time)
{
    int i;
    for (i = 0; i < n; ++i, m += 4) {
        vec3 p = beacon(m, time);
        const float *c = i == active ? LAMP_ACTIVE : LAMP_IDLE;
        float col[3];
        col[0] = (float)sqrt(c[0]); col[1] = (float)sqrt(c[1]); col[2] = (float)sqrt(c[2]);
        set_material(r, ARC_MAT_GLOW, col, 0.0);
        glPushMatrix();
        glTranslated(p.x, p.y, p.z);
        glScaled(0.07, 0.07, 0.07);
        glCallList(r->list_sphere);
        glPopMatrix();
        if (m[3] != r->ring_radius) {
            glNewList(r->list_ring, GL_COMPILE);
            mesh_ring(v3(0, 0, 0), v3(0, 0, 1), m[3], 0.012);
            glEndList();
            r->ring_radius = m[3];
        }
        glPushMatrix();
        glTranslated(m[0], m[1], m[2] + 0.01);
        glCallList(r->list_ring);
        glPopMatrix();
    }
}

/* Camera-facing quad of half size s at p; uv in [-1, 1]. */
static void billboard(vec3 p, vec3 right, vec3 up, double s)
{
    glTexCoord2d(-1, -1);
    glVertex3d(p.x - s * (right.x + up.x), p.y - s * (right.y + up.y), p.z - s * (right.z + up.z));
    glTexCoord2d(1, -1);
    glVertex3d(p.x + s * (right.x - up.x), p.y + s * (right.y - up.y), p.z + s * (right.z - up.z));
    glTexCoord2d(1, 1);
    glVertex3d(p.x + s * (right.x + up.x), p.y + s * (right.y + up.y), p.z + s * (right.z + up.z));
    glTexCoord2d(-1, 1);
    glVertex3d(p.x - s * (right.x - up.x), p.y - s * (right.y - up.y), p.z - s * (right.z - up.z));
}

/* Halos around glowing parts and checkpoint beacons, additive. */
static void draw_glow(const arc_renderer *r, const arc_world *w, vec3 eye, mat4 view,
                      const double *m, int n, int active, double time)
{
    vec3 right = v3(view.m[0], view.m[4], view.m[8]), up = v3(view.m[1], view.m[5], view.m[9]);
    int i;
    p_glUseProgram(r->prog_glow);
    p_glUniform3f(r->glow_cam, (float)eye.x, (float)eye.y, (float)eye.z);
    p_glUniform2f(r->glow_fog, FOG_DENSITY, (float)v3_len(v3_sub(eye, r->cam_target)));
    p_glUniform1f(r->glow_beam, 0.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);
    glBegin(GL_QUADS);
    for (i = 0; i < w->n_visuals; ++i) {
        const arc_visual *v = &w->visuals[i];
        vec3 c, toward;
        double s;
        if (v->material != ARC_MAT_GLOW)
            continue;
        c = visual_center(w, v);
        s = v->type == ARC_VIS_RING ? 1.7 * (v->radius + v->radius2) : 3.5 * v->radius;
        toward = v3_normalize(v3_sub(eye, c));
        glColor4f(v->color[0], v->color[1], v->color[2], v->type == ARC_VIS_RING ? 0.4f : 0.3f);
        billboard(v3_madd(c, toward, 0.6 * s), right, up, s);
    }
    for (i = 0; i < n; ++i) {
        const float *c = i == active ? LAMP_ACTIVE : LAMP_IDLE;
        vec3 p = beacon(m + 4 * i, time);
        glColor4f(c[0], c[1], c[2], 0.8f);
        billboard(v3_madd(p, v3_normalize(v3_sub(eye, p)), 0.3), right, up, 0.45);
    }
    glEnd();
    /* beacons: vertical beams of light, facing the camera */
    p_glUniform1f(r->glow_beam, 1.0f);
    glBegin(GL_QUADS);
    for (i = 0; i < n; ++i) {
        const float *c = i == active ? LAMP_ACTIVE : LAMP_IDLE;
        vec3 base = v3(m[4 * i], m[4 * i + 1], m[4 * i + 2]);
        vec3 side = v3_normalize(v3_cross(v3(0, 0, 1), v3_sub(eye, base)));
        double s = 0.1, top = 3.5;
        glColor4f(c[0], c[1], c[2], 0.5f);
        glTexCoord2d(-1, 0); glVertex3d(base.x - s * side.x, base.y - s * side.y, base.z);
        glTexCoord2d(1, 0); glVertex3d(base.x + s * side.x, base.y + s * side.y, base.z);
        glTexCoord2d(1, 1); glVertex3d(base.x + s * side.x, base.y + s * side.y, base.z + top);
        glTexCoord2d(-1, 1); glVertex3d(base.x - s * side.x, base.y - s * side.y, base.z + top);
    }
    glEnd();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

static void fullscreen(void)
{
    glBegin(GL_QUADS);
    glVertex2d(-1, -1); glVertex2d(1, -1); glVertex2d(1, 1); glVertex2d(-1, 1);
    glEnd();
}

/* The modelview matrix stays the identity between draws, and current. */
static void set_projection(mat4 m)
{
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(m.m);
    glMatrixMode(GL_MODELVIEW);
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
#define LOAD_FN(type, name) LOAD_GL(p_##name, #name); ok = ok && p_##name;

static int load_gl(void)
{
    int ok = 1;
    GL_FUNCS(LOAD_FN)
    return ok;
}

/* Renderbuffer attached to the bound framebuffer (samples = 0: single-sampled). */
static GLuint attach(GLenum attachment, GLenum format, int samples, int w, int h)
{
    GLuint rb;
    p_glGenRenderbuffers(1, &rb);
    p_glBindRenderbuffer(GL_RENDERBUFFER, rb);
    p_glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, format, w, h);
    p_glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, rb);
    return rb;
}

static int create_fbo(arc_renderer *r)
{
    GLint max_samples = 0;
    int samples, ok;
    glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
    samples = max_samples < MSAA ? max_samples : MSAA;
    p_glGenFramebuffers(1, &r->resolve_fbo);
    p_glBindFramebuffer(GL_FRAMEBUFFER, r->resolve_fbo);
    r->resolve_rb = attach(GL_COLOR_ATTACHMENT0, GL_RGBA8, 0, r->w, r->h);
    ok = p_glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    p_glGenFramebuffers(1, &r->fbo);
    p_glBindFramebuffer(GL_FRAMEBUFFER, r->fbo);
    r->color_rb = attach(GL_COLOR_ATTACHMENT0, GL_RGBA8, samples, r->w, r->h);
    r->depth_rb = attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, samples, r->w, r->h);
    return ok && p_glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
}

/* Depth texture with hardware comparison (2x2 PCF with linear filtering). */
static int create_shadow_map(arc_renderer *r)
{
    int ok;
    glGenTextures(1, &r->shadow_tex);
    glBindTexture(GL_TEXTURE_2D, r->shadow_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_SIZE, SHADOW_SIZE, 0,
                 GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    p_glGenFramebuffers(1, &r->shadow_fbo);
    p_glBindFramebuffer(GL_FRAMEBUFFER, r->shadow_fbo);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, r->shadow_tex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    ok = p_glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return ok;
}

arc_renderer *arc_render_create(int width, int height, int visible)
{
    arc_renderer *r = (arc_renderer *)calloc(1, sizeof(arc_renderer));
    int msaa;
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
    /* windows get multisampling if available; offscreen frames draw into the FBO */
    for (msaa = visible ? MSAA : 0; !r->win && msaa >= 0; msaa -= MSAA) {
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, msaa > 0);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, msaa);
        r->win = SDL_CreateWindow("arc", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width,
                                  height, SDL_WINDOW_OPENGL | (visible ? SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
                                                                       : SDL_WINDOW_HIDDEN));
    }
    if (!r->win || !(r->ctx = SDL_GL_CreateContext(r->win))) {
        arc_render_destroy(r);
        return 0;
    }
    SDL_GL_MakeCurrent(r->win, r->ctx);
    if (!load_gl() || !build_programs(r) || !create_shadow_map(r)) {
        arc_render_destroy(r);
        return 0;
    }
    if (visible) {
        SDL_GL_SetSwapInterval(1);
        SDL_GL_GetDrawableSize(r->win, &r->w, &r->h);
    } else if (!create_fbo(r)) {
        arc_render_destroy(r);
        return 0;
    }

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);
    r->list_sphere = glGenLists(3);
    r->list_terrain = r->list_sphere + 1;
    r->list_ring = r->list_sphere + 2;
    r->list_visuals = glGenLists(ARC_MAX_VISUALS);
    glNewList(r->list_sphere, GL_COMPILE);
    unit_sphere();
    glEndList();
    return r;
}

void arc_render_destroy(arc_renderer *r)
{
    if (!r)
        return;
    if (r->ctx) {
        SDL_GL_MakeCurrent(r->win, r->ctx);
        if (r->list_sphere)
            glDeleteLists(r->list_sphere, 3);
        if (r->list_visuals)
            glDeleteLists(r->list_visuals, ARC_MAX_VISUALS);
        if (p_glDeleteProgram) {
            if (r->prog_scene) p_glDeleteProgram(r->prog_scene);
            if (r->prog_glow) p_glDeleteProgram(r->prog_glow);
            if (r->prog_post) p_glDeleteProgram(r->prog_post);
        }
        if (r->shadow_tex)
            glDeleteTextures(1, &r->shadow_tex);
        if (r->shadow_fbo)
            p_glDeleteFramebuffers(1, &r->shadow_fbo);
        if (r->fbo) {
            p_glDeleteFramebuffers(1, &r->fbo);
            p_glDeleteRenderbuffers(1, &r->color_rb);
            p_glDeleteRenderbuffers(1, &r->depth_rb);
        }
        if (r->resolve_fbo) {
            p_glDeleteFramebuffers(1, &r->resolve_fbo);
            p_glDeleteRenderbuffers(1, &r->resolve_rb);
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

/* Light-space matrix of the shadow map, a square around the camera target
 * snapped to whole texels so that shadow edges do not shimmer. */
static mat4 shadow_matrix(vec3 center)
{
    vec3 dir = v3_normalize(v3(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2])), c;
    mat4 view = m4_look_at(v3(0, 0, 0), v3_neg(dir), v3(0, 0, 1));
    double texel = 2.0 * SHADOW_RANGE / SHADOW_SIZE;
    c = m4_point(view, center);
    c.x = floor(c.x / texel) * texel;
    c.y = floor(c.y / texel) * texel;
    return m4_mul(m4_ortho(c.x - SHADOW_RANGE, c.x + SHADOW_RANGE, c.y - SHADOW_RANGE,
                           c.y + SHADOW_RANGE, -c.z - 30.0, -c.z + 30.0), view);
}

static void set_scene_uniforms(const arc_renderer *r, const arc_world *w, vec3 eye, mat4 shadow,
                               const double *markers, int n_markers, int active)
{
    const scene_uniforms *u = &r->su;
    vec3 eye_pos = v3(0, 0, -100), eye_dir = v3(1, 0, 0), lamp = v3(0, 0, -100);
    float eye_col[3] = { 0, 0, 0 }, lamp_col[3] = { 0, 0, 0 }, m[16];
    mat4 bias;
    int i;

    /* the robot's eye: the first glowing ring on the body, shining forwards
     * and a little downwards */
    for (i = 0; i < w->n_visuals; ++i) {
        const arc_visual *v = &w->visuals[i];
        if (v->material == ARC_MAT_GLOW && v->type == ARC_VIS_RING && v->body == 0) {
            const arc_body *b = &w->bodies[0];
            vec3 axis = q_rotate(b->q, v3_normalize(v->p1));
            eye_pos = v3_madd(visual_center(w, v), axis, 0.05);
            eye_dir = v3_normalize(v3_add(axis, v3(0, 0, -0.35)));
            eye_col[0] = EYE_LIGHT * v->color[0] * v->color[0];
            eye_col[1] = EYE_LIGHT * v->color[1] * v->color[1];
            eye_col[2] = EYE_LIGHT * v->color[2] * v->color[2];
            break;
        }
    }
    if (markers && n_markers > 0) {
        const float *c = active == 0 ? LAMP_ACTIVE : LAMP_IDLE;
        lamp = v3(markers[0], markers[1], markers[2] + BEACON_HEIGHT);
        for (i = 0; i < 3; ++i)
            lamp_col[i] = (active == 0 ? 1.6f : 0.8f) * c[i];
    }

    memset(&bias, 0, sizeof(bias));
    bias.m[0] = bias.m[5] = bias.m[10] = bias.m[12] = bias.m[13] = bias.m[14] = 0.5;
    bias.m[15] = 1.0;
    shadow = m4_mul(bias, shadow);
    for (i = 0; i < 16; ++i)
        m[i] = (float)shadow.m[i];

    p_glUniform3f(u->cam, (float)eye.x, (float)eye.y, (float)eye.z);
    p_glUniform2f(u->fog, FOG_DENSITY, (float)v3_len(v3_sub(eye, r->cam_target)));
    p_glUniform1f(u->time, (float)w->time);
    p_glUniform2f(u->focus, (float)r->cam_target.x, (float)r->cam_target.y);
    p_glUniformMatrix4fv(u->shadow_mat, 1, GL_FALSE, m);
    p_glUniform3f(u->eye_pos, (float)eye_pos.x, (float)eye_pos.y, (float)eye_pos.z);
    p_glUniform3f(u->eye_dir, (float)eye_dir.x, (float)eye_dir.y, (float)eye_dir.z);
    p_glUniform3fv(u->eye_col, 1, eye_col);
    p_glUniform3f(u->lamp_pos, (float)lamp.x, (float)lamp.y, (float)lamp.z);
    p_glUniform3fv(u->lamp_col, 1, lamp_col);
}

int arc_render_frame(arc_renderer *r, const arc_sim *s, int world,
                     const double *markers, int n_markers, int active,
                     unsigned char *rgb)
{
    const arc_world *w = arc_sim_world(s, world);
    const arc_terrain *t = &w->terrain;
    double yaw, near_plane = 0.05, far_plane = 120.0, aspect, top;
    mat4 view, proj, shadow;
    vec3 eye;

    handle_events(r);
    if (r->closed)
        return 0;
    SDL_GL_MakeCurrent(r->win, r->ctx);

    update_terrain(r, t);
    update_visuals(r, w);
    update_camera(r, w);

    yaw = r->cam_heading + ARC_PI + r->cam_yaw;
    eye = v3_add(r->cam_target, v3(r->cam_dist * cos(r->cam_pitch) * cos(yaw),
                                   r->cam_dist * cos(r->cam_pitch) * sin(yaw),
                                   r->cam_dist * sin(r->cam_pitch)));
    /* keep the eye above the terrain */
    eye.z = arc_max(eye.z, arc_terrain_height(t, eye.x, eye.y) + 0.2);
    view = m4_look_at(eye, r->cam_target, v3(0, 0, 1));
    aspect = (double)r->w / (r->h > 0 ? r->h : 1);
    top = near_plane * tan(0.5 * 50.0 * ARC_PI / 180.0);
    proj = m4_frustum(-top * aspect, top * aspect, -top, top, near_plane, far_plane);
    shadow = shadow_matrix(r->cam_target);

    /* shadow pass: terrain and robot, depth only */
    p_glBindFramebuffer(GL_FRAMEBUFFER, r->shadow_fbo);
    glViewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
    glClear(GL_DEPTH_BUFFER_BIT);
    p_glUseProgram(0);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.5f, 4.0f);
    set_projection(shadow);
    glCallList(r->list_terrain);
    draw_robot(r, w, 0);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    p_glBindFramebuffer(GL_FRAMEBUFFER, r->fbo);
    glViewport(0, 0, r->w, r->h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* scene */
    set_projection(m4_mul(proj, view));
    p_glUseProgram(r->prog_scene);
    p_glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, r->shadow_tex);
    set_scene_uniforms(r, w, eye, shadow, markers, n_markers, active);
    p_glUniform4f(r->su.mat, 0.25f, 16.0f, 0.0f, 1.0f);
    p_glUniform4f(r->su.aux, (float)t->ox, (float)t->oy, (float)t->cell, 0.0f);
    glCallList(r->list_terrain);
    draw_robot(r, w, 1);
    if (markers && n_markers > 0)
        draw_markers(r, markers, n_markers, active, w->time);

    /* glow */
    draw_glow(r, w, eye, view, markers, markers ? n_markers : 0, active, w->time);

    /* vignette */
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_DST_COLOR, GL_ZERO);
    p_glUseProgram(r->prog_post);
    p_glUniform2f(r->post_size, (float)r->w, (float)r->h);
    fullscreen();
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    p_glUseProgram(0);

    if (rgb) {
        /* RGBA is the driver's fast path (RGB gets converted pixel by pixel
         * in the driver). Flip the rows and drop alpha here, copying 4 bytes
         * per pixel of which the next pixel overwrites the last. */
        int width = r->w, height = r->h, row, x;
        if (!r->scratch)
            r->scratch = (unsigned char *)malloc((size_t)width * height * 4);
        if (r->fbo) {
            p_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, r->resolve_fbo);
            p_glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT,
                                GL_NEAREST);
            p_glBindFramebuffer(GL_READ_FRAMEBUFFER, r->resolve_fbo);
        }
        glReadBuffer(r->fbo ? GL_COLOR_ATTACHMENT0 : GL_BACK);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, r->scratch);
        for (row = 0; row < height; ++row) {
            const unsigned char *src = r->scratch + (size_t)(height - 1 - row) * width * 4;
            unsigned char *dst = rgb + (size_t)row * width * 3;
            for (x = 0; x < width - 1; ++x)
                memcpy(dst + 3 * x, src + 4 * x, 4);
            memcpy(dst + 3 * x, src + 4 * x, 3);
        }
    }
    if (r->visible)
        SDL_GL_SwapWindow(r->win);
    return 1;
}
