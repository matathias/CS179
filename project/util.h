/* CS/CNS 171 Fall 2014
 *
 * This header file contains some useful utility functions.
 */

#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

struct Point_Light;
struct Material;
struct Object;
struct Pixel;

/* Returns -1 for negative numbers, 1 for positive numbers, and 0 for zero. */
int sign(double s);
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

/* Implicit Superquadric function. */
// vec is a 3-vector
double isq(double *vec, double e, double n);

/* Ray Equation */
// a and b are both 3-vectors
void findRay(double *a, double *b, double *c, double t);

/* Apply the Inverse Transform to a to get a new, usable a. */
// unScale and unRotate are 3x3 matrices. a and newA are 3-vectors
void newa(double *unScale, double *unRotate, double *a, double *newA);
/* Apply the Inverse Transform to b to get a new, usable b. */
// unScale and unRotate are 3x3 matrices. unTranslate, b, and newB are 3-vectors
void newb(double *unScale, double *unRotate, double *unTranslate, double *b, 
          double *newB);

/* Finds the scalar coefficients of the quadratic equation with the two given
 * vectors. If positiveb is true then the returned coeffs will all be multiplied
 * by -1 if b is negative, to ensure that b is positive. */
 // a, b, and c are 3-vectors
void findCoeffs(double *a, double *b, double *c, bool positiveb);
/* Finds the roots of the quadratic with the coefficients specified by the input
 * Vector3d. If one of the roots is complex then FLT_MAX is returned instead. */
// coeffs is a 3-vector, roots is a 2-vector
void findRoots(double *coeffs, double *roots);

/* Uses Newton's method to find the t value at which a ray hits the superquadric.
 * If the ray actually misses the superquadric then FLT_MAX is returned instead.*/
// a and b are 3-vectors
double updateRule(double *a, double *b, double e, double n, double t, 
                  double epsilon);


/* Gradient of the isq function. */
// vec and grad are 3-vectors
void isqGradient(double *vec, double *grad, double e, double n);
/* Derivative of the isq function. */
// vec and a are 3-vectors
double gPrime(double *vec, double *a, double e, double n);
/* Unit normal vector at a point on the superquadric */
// r is a 3x3 matrix
// vec1, vec2, and un are 3-vectors
void unitNormal(double *r, double *vec1, double *vec2, double *un, double tt, 
                double e, double n);

// Simple function to convert an angle in degrees to radians
double deg2rad(double angle);
// Simple function to convert an angle in radians to degrees
double rad2deg(double angle);

// Returns the angle between two vectors.
// Both a and b are 3-vectors.
double vectorAngle(double *a, double *b);

// Calculates the refracted ray from an input ray and normal and a snell ratio
// If there is total internal reflection, then a vector of FLT_MAX is returned
// instead.
// a, n, and ref are 3-vectors
void refractedRay(double *a, double *n, double *ref, double snell);

