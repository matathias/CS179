/* CS/CNS 171 Fall 2014
 *
 * This file contains the actual implementations of the functions in util.h.
 */

#include "util.h"
#include <math.h>
#include <float.h>

/* Returns -1 for negative numbers, 1 for positive numbers, and 0 for zero. */
int sign(double s)
{
    if(s > 0) return 1;
    if(s < 0) return -1;
    return 0;
}

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

void findFilmA(double x, double y, double *e1, double *e2, double *e3, double *film)
{
    for (int i = 0; i < 3; i++) {
        film[i] = (filmDepth * e3[i]) + (x * e1[i]) + (y * e2[i]);
    }
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

/* Implicit Superquadric function. */
// vec is a 3-vector
double isq(double *vec, double e, double n)
{
    // Test for n = 0 now to prevent divide-by-zero errors.
    if (n == 0)
        return FLT_MAX;
    
    double zTerm = pow(pow(vec[2], 2.0), 1.0 / (double) n);

    // Test for e = 0 now to prevent divide-by-zero errors.
    if (e == 0)
        return zTerm;

    double xTerm = pow(pow(vec[0], 2.0), 1.0 / (double) e);
    double yTerm = pow(pow(vec[1], 2.0), 1.0 / (double) e);
    double xyTerm = pow(xTerm + yTerm, e / (double) n);
    return xyTerm + zTerm - 1.0;
}

/* Ray Equation */
// a and b are both 3-vectors
void findRay(double *a, double *b, double *c, double t)
{
    c[0] = (a[0] * t) + b[0];
    c[1] = (a[1] * t) + b[1];
    c[2] = (a[2] * t) + b[2];
}

/* Apply the Inverse Transform to a to get a new, usable a. */
// unScale and unRotate are 3x3 matrices. a and newA are 3-vectors
void newa(double *unScale, double *unRotate, double *a, double *newA)
{
    newA[0] = (unRotate[0] * a[0]) + (unRotate[1] * a[1]) + (unRotate[2] * a[2]);
    newA[1] = (unRotate[3] * a[0]) + (unRotate[4] * a[1]) + (unRotate[5] * a[2]);
    newA[2] = (unRotate[6] * a[0]) + (unRotate[7] * a[1]) + (unRotate[8] * a[2]);
    
    double a0 = newA[0];
    double a1 = newA[1];
    double a2 = newA[2];
    newA[0] = (unScale[0] * a0) + (unScale[1] * a1) + (unScale[2] * a2);
    newA[1] = (unScale[3] * a0) + (unScale[4] * a1) + (unScale[5] * a2);
    newA[2] = (unScale[6] * a0) + (unScale[7] * a1) + (unScale[8] * a2);
}

/* Apply the Inverse Transform to b to get a new, usable b. */
// unScale and unRotate are 3x3 matrices. unTranslate, b, and newB are 3-vectors
void newb(double *unScale, double *unRotate, double *unTranslate, double *b, 
          double *newB)
{
    // b + unTranslate
    newB[0] = b[0] + unTranslate[0];
    newB[1] = b[1] + unTranslate[1];
    newB[2] = b[2] + unTranslate[2];
    
    double b0 = newB[0];
    double b1 = newB[1];
    double b2 = newB[2];
    
    // unRotate * (b + unTranslate)
    newB[0] = (unRotate[0] * b0) + (unRotate[1] * b1) + (unRotate[2] * b2);
    newB[1] = (unRotate[3] * b0) + (unRotate[4] * b1) + (unRotate[5] * b2);
    newB[2] = (unRotate[6] * b0) + (unRotate[7] * b1) + (unRotate[8] * b2);
    
    b0 = newB[0];
    b1 = newB[1];
    b2 = newB[2];
    
    // unScale * (unRotate * (b + unTranslate))
    newB[0] = (unScale[0] * b0) + (unScale[1] * b1) + (unScale[2] * b2);
    newB[1] = (unScale[3] * b0) + (unScale[4] * b1) + (unScale[5] * b2);
    newB[2] = (unScale[6] * b0) + (unScale[7] * b1) + (unScale[8] * b2);
}

/* Finds the scalar coefficients of the quadratic equation with the two given
 * vectors. If positiveb is true then the returned coeffs will all be multiplied
 * by -1 if b is negative, to ensure that b is positive. */
 // a, b, and c are 3-vectors
void findCoeffs(double *a, double *b, double *c, bool positiveb)
{
    c[0] = dot(a, a);
    c[1] = 2 * dot(a, b);
    c[2] = dot(b, b) - 3;
    
    if (positiveb)
    {
        if (c[1] < 0){
            c[0] *= -1;
            c[1] *= -1;
            c[2] *= -1;
        }
    }
}

/* Finds the roots of the quadratic with the coefficients specified by the input
 * Vector3d. If one of the roots is complex then FLT_MAX is returned instead. */
// coeffs is a 3-vector, roots is a 2-vector
void findRoots(double *coeffs, double *roots)
{
    double interior = pow(coeffs[1], 2) - (4 * coeffs[0] * coeffs[2]);
    if (interior < 0)
    {
        roots[0] = FLT_MAX;
        roots[1] = FLT_MAX;
    }
    else
    {
        roots[0] = (-coeffs[1] - sqrt(interior)) / (double) (2 * coeffs[0]);
        roots[1] = (2 * coeffs[2]) / (double) (-coeffs[1] - sqrt(interior));
    }
}

/* Uses Newton's method to find the t value at which a ray hits the superquadric.
 * If the ray actually misses the superquadric then FLT_MAX is returned instead.*/
// a and b are 3-vectors
double updateRule(double *a, double *b, double e, double n, double t, double epsilon)
{
    double vec[3];
    findRay(a, b, &vec, t);
    double gP = gPrime(&vec, a, e, n);
    double gPPrevious = gP;
    double g = 0.0;
    double tnew = t, told = t;
    bool stopPoint = false;

    while (!stopPoint)
    {
        told = tnew;
        findRay(a, b, &vec, told);
        gP = gPrime(&vec, a, e, n);
        g = isq(&vec, e, n);

        if ((g - epsilon) <= 0)
        {
            stopPoint = true;
        }
        else if (sign(gP) != sign(gPPrevious) || gP == 0)
        {
            stopPoint = true;
            tnew = FLT_MAX;
        }
        else
        {
            tnew = told - (g / gP);
            gPPrevious = gP;
        }
    }
    
    delete[] vec;

    return tnew;
}

/* Gradient of the isq function. */
// vec and grad are 3-vectors
void isqGradient(double *vec, double *grad, double e, double n)
{
    double xval = 0.0, yval = 0.0, zval = 0.0;
    // Check for n = 0 to prevent divide-by-zero errors
    if (n == 0)
    {
        cout << "n is zero!" << endl;
        xval = yval = zval = FLT_MAX;
    }
    // Check for e = 0 to prevent divide-by-zero errors
    else if (e == 0)
    {
        cout << "e is  zero!" << endl;
        xval = yval = FLT_MAX;
        zval = (2 * vec[2] * pow(pow(vec[2], 2), ((double) 1 / n) - 1)) / (double) n;
    }
    else
    {
        double xterm = pow(pow(vec[0], 2.0), (double) 1 / e);
        double yterm = pow(pow(vec[1], 2.0), (double) 1 / e);
        double xyterm = pow(xterm + yterm, ((double) e / n) - 1);
        double x2term = (2 * vec[0] * pow(pow(vec[0], 2.0), ((double) 1 / e) - 1));
        double y2term = (2 * vec[1] * pow(pow(vec[1], 2.0), ((double) 1 / e) - 1));
        xval = x2term * xyterm / (double) n;
        yval = y2term * xyterm / (double) n;
        zval = (2 * vec[2] * pow(pow(vec[2], 2.0), ((double) 1 / n) - 1)) / (double) n;
    }
    
    grad[0] = xval;
    grad[1] = yval;
    grad[2] = zval;
}

/* Derivative of the isq function. */
// vec and a are 3-vectors
double gPrime(double *vec, double *a, double e, double n)
{
    double tmp[3];
    isqGradient(vec, &tmp, e, n);
    double val = dot(a, &tmp);
    delete[] tmp;
    return val;
}

/* Unit normal vector at a point on the superquadric */
// r is a 3x3 matrix
// vec1, vec2, and un are 3-vectors
void unitNormal(double *r, double *vec1, double *vec2, double *un, double tt, double e, double n)
{
    findRay(vec1, vec2, un, tt);
    isqGradient(un, un, e, n);
    
    double un0 = un[0];
    double un1 = un[1];
    double un2 = un[2];
    nor = r * nor;
    nor.normalize();
    return nor;
}

// Simple function to convert an angle in degrees to radians
double deg2rad(double angle)
{
    return angle * M_PI / 180.0;
}
// Simple function to convert an angle in radians to degrees
double rad2deg(double angle)
{
    return angle * (180.0 / M_PI);
}

// Returns the angle between two vectors.
// Both a and b are 3-vectors.
double vectorAngle(double *a, double *b)
{
    double dot = dot(a, b);
    double mag = norm(a) * norm(b);

    return acos(dot / (double) mag);
}

// Calculates the refracted ray from an input ray and normal and a snell ratio
// If there is total internal reflection, then a vector of FLT_MAX is returned
// instead.
// a, n, and ref are 3-vectors
void refractedRay(double *a, double *n, double *ref, double snell)
{
    double tmp = dot(n, a);
    n[0] *= -1;
    n[1] *= -1;
    n[2] *= -1;
    double cos1 = dot(n, a);
    if (cos1 < 0)
    {
        cos1 = tmp;
    }
    else {
        n[0] *= -1;
        n[1] *= -1;
        n[2] *= -1;
    }
    double radicand = 1 - (pow(snell, 2) * (1 - pow(cos1,2)));

    if (radicand < 0)
    {
        ref[0] = FLT_MAX;
        ref[1] = FLT_MAX;
        ref[2] = FLT_MAX;
    }
    else
    {
        double cos2 = sqrt(radicand);

        ref[0] = (snell * a[0]) + (((snell * cos1) - cos2) * n[0]);
        ref[1] = (snell * a[1]) + (((snell * cos1) - cos2) * n[1]);
        ref[2] = (snell * a[2]) + (((snell * cos1) - cos2) * n[2]);
    }
}
