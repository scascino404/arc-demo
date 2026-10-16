/*
 * arc_robot.h - the quadruped: a torso and four 3-joint legs
 * (hip abduction, hip flexion, knee). Forward is +X, up is +Z.
 *
 * Bodies: 0 torso, then hip, thigh, shin for each leg in the order
 * front-left, front-right, hind-left, hind-right. Joint 3*leg + k drives
 * body 3*leg + k + 1.
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
} arc_robot_info;

/* Builds the robot into an empty world (posed in the default stance, torso
 * at stand height above z = 0) and fills info. */
void arc_robot_build(arc_world *w, arc_robot_info *info);

#endif
