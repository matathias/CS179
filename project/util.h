/*
 * This header file contains some useful utility functions.
 */

#pragma once

#include <iostream>
#include <vector>

using namespace std;

struct Point_Light;
struct Material;
struct Object;
struct Pixel;

/* Returns the norm of the given vector. */
double norm(double *vec);
/* Normalizes the given vector. */
void normalize(double *vec);
/* Returns the dot product of the given vectors. */
double dot(double *a, double *b);
/* Returns the cross product a x b into vector c. */
void cross(double *a, double *b, double *c);

/* Gets the rotation matrix for a given rotation axis (x, y, z) and angle. */
void get_rotate_mat(double x, double y, double z, double angle, double *m);
/* Gets the scaling matrix for a given scaling vector (x, y, z). */
void get_scale_mat(double x, double y, double z, double *m);
