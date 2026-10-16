/*
 * arc_robot.h - the robot: a "leaper"-style spider. A spherical body with a
 * glowing eye marking the front (+X) hangs below four long legs splayed
 * diagonally outwards, each with a hip yaw, a hip pitch and a knee joint; in
 * the default stance the knees are high above the body and the long shins
 * come down steeply. Up is +Z.
 *
 * Bodies: 0 body, then coxa, femur, tibia for each leg in the order
 * front-left, front-right, hind-left, hind-right. Joint 3*leg + k drives
 * body 3*leg + k + 1. Positive angles: yaw swings the leg forwards, pitch
 * and knee lift the distal segment.
 */
#ifndef ARC_ROBOT_H
#define ARC_ROBOT_H

#include "arc_phys.h"

#define ARC_ROBOT_LEGS 4
#define ARC_ROBOT_JOINTS 12

typedef struct {
    int foot_sphere[ARC_ROBOT_LEGS];
    int n_torso_spheres;           /* spheres [0, n) belong to the torso */
    double default_angles[ARC_ROBOT_JOINTS];
    double lo[ARC_ROBOT_JOINTS], hi[ARC_ROBOT_JOINTS];
    double stand_height;           /* torso height above flat ground in the default pose */
    double total_mass;
    double footprint;              /* horizontal distance from the torso center to the feet */
} arc_robot_info;

/* Builds the robot into an empty world (posed in the default stance, torso
 * at stand height above z = 0) and fills info. */
void arc_robot_build(arc_world *w, arc_robot_info *info);

#endif
