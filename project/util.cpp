/*
 * This file contains the actual implementations of the functions in util.h.
 */

#include "util.h"
#include <math.h>
#include <float.h>

/* Returns the norm of the given vector. */
double norm(double *vec)
{
    double n = 0;
    for (int i = 0; i < 3; i++) {
        n += vec[i];
    }
    return sqrt(n);
}

/* Normalizes the given vector. */
void normalize(double *vec)
{
    double n = norm(vec);
    for (int i = 0; i < 3; i++) {
        vec[i] = vec[i] / n;
    }
}

/* Returns the dot product of the given vectors. */
double dot(double *a, double *b)
{
    double d = 0;
    for (int i = 0; i < 3; i++) {
        d += a[i] * b[i];
    }
    return d;
}

/* Returns the cross product a x b into vector c. */
void cross(double *a, double *b, double *c)
{
    c[0] = (a[1] * b[2]) - (a[2] * b[1]);
    c[1] = (a[2] * b[0]) - (a[0] * b[2]);
    c[2] = (a[0] * b[1]) - (a[1] * b[0]);
}

/* Gets the rotation matrix for a given rotation axis (x, y, z) and angle. */
void get_rotate_mat(double x, double y, double z, double angle, double *m)
{
    double nor = sqrt((x * x) + (y * y) + (z * z));
    x = x / nor;
    y = y / nor;
    z = z / nor;
    angle = deg2rad(angle);

    m[0] = pow(x,2) + (1 - pow(x,2)) * cos(angle);
    m[1] = (x * y * (1 - cos(angle))) - (z * sin(angle));
    m[2] = (x * z * (1 - cos(angle))) + (y * sin(angle));

    m[3] = (y * x * (1 - cos(angle))) + (z * sin(angle));
    m[4] = pow(y,2) + (1 - pow(y,2)) * cos(angle);
    m[5] = (y * z * (1 - cos(angle))) - (x * sin(angle));

    m[6] = (z * x * (1 - cos(angle))) - (y * sin(angle));
    m[7] = (z * y * (1 - cos(angle))) + (x * sin(angle));
    m[8] = pow(z,2) + (1 - pow(z,2)) * cos(angle);
}

/* Gets the scaling matrix for a given scaling vector (x, y, z). */
void get_scale_mat(double x, double y, double z, double *m)
{
    m[0] = x;
    m[1] = 0;
    m[2] = 0;
    
    m[3] = 0;
    m[4] = y;
    m[5] = 0;
    
    m[6] = 0;
    m[7] = 0;
    m[8] = z;
}
