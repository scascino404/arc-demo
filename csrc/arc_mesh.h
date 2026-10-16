/*
 * arc_mesh.h - geometry of the rendered scene, emitted with immediate mode
 * OpenGL for the renderer to compile into display lists: the primitives of
 * the robot's visuals (in their body frame) and the block terrain (in
 * world space). Face culling is off, so winding does not matter.
 */
#ifndef ARC_MESH_H
#define ARC_MESH_H

#include "arc_terrain.h"

/* Sphere of radius 1 around the origin. */
void arc_mesh_sphere(void);

/* Closed cylinder along p0-p1; ribbed ones get a corrugated surface. */
void arc_mesh_cylinder(vec3 p0, vec3 p1, double rad, int ribbed);

/* Faceted prism: a rectangle of half extents (w, aspect w) with chamfered
 * corners, w going from r0 at p0 to r1 at p1. */
void arc_mesh_taper(vec3 p0, vec3 p1, double r0, double r1, double aspect);

/* Box with rounded edges, center c and half extents e. */
void arc_mesh_box(vec3 c, vec3 e);

/* Torus around axis through c: major radius and tube radius. */
void arc_mesh_ring(vec3 c, vec3 axis, double major, double tube);

/* Column tops, shaded by ambient occlusion, and column sides down to the
 * lower neighbour, but no lower than z = bottom (the void around the grid). */
void arc_mesh_terrain(const arc_terrain *t, double bottom);

#endif
