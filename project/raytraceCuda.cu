#include <cassert>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "raytraceCuda.cuh"

struct Point_Light
{
    double *position;    //3-vector
    double *color;       //3-vector
    double attenuation_k;
};

struct Material
{
    double *diffuse;     //3-vector
    double *ambient;     //3-vector
    double *specular;    //3-vector
    double shine;
    double snell;
    double opacity;
};

struct Object
{
    double e;
    double n;
    Material *mat;
    double *scale;          //3x3-matrix
    double *unScale;        //3x3-matrix
    double *rotate;         //3x3-matrix
    double *unRotate;       //3x3-matrix
    double *translate;      //3-vector
    double *unTranslate;    //3-vector
};

struct Pixel
{
    double red;
    double green;
    double blue;
};
               
/********** Helper Functions **************************************************/
__device__
void cProduct(double *a, double *b, double *c) 
{
    for (int i = 0; i < 3; i++) {
        c[i] = a[i] * b[i];
    }
}
__device__
void cWiseMin(double *a, double *b, double *out)
{
    for (int i = 0; i < 3; i++) {
        if (a[i] < b[i])
            out[i] = a[i];
        else
            out[i] = b[i];
    }
}

__device__
void findFilmA(double x, double y, double *e1, double *e2, double *e3, 
               double filmDepth, double *film)
{
    for (int i = 0; i < 3; i++) {
        film[i] = (filmDepth * e3[i]) + (x * e1[i]) + (y * e2[i]);
    }
}

/* Returns -1 for negative numbers, 1 for positive numbers, and 0 for zero. */
__device__
int sign(double s)
{
    if(s > 0) return 1;
    if(s < 0) return -1;
    return 0;
}

/* Returns the norm of the given vector. */
__device__
double norm(double *vec)
{
    double n = 0;
    for (int i = 0; i < 3; i++) {
        n += vec[i];
    }
    return sqrt(n);
}

/* Normalizes the given vector. */
__device__
void normalize(double *vec)
{
    double n = norm(vec);
    for (int i = 0; i < 3; i++) {
        vec[i] = vec[i] / n;
    }
}

/* Returns the dot product of the given vectors. */
__device__
double dot(double *a, double *b)
{
    double d = 0;
    for (int i = 0; i < 3; i++) {
        d += a[i] * b[i];
    }
    return d;
}

/* Implicit Superquadric function. */
// vec is a 3-vector
__device__
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
__device__
void findRay(double *a, double *b, double *c, double t)
{
    c[0] = (a[0] * t) + b[0];
    c[1] = (a[1] * t) + b[1];
    c[2] = (a[2] * t) + b[2];
}

/* Apply the Inverse Transform to a to get a new, usable a. */
// unScale and unRotate are 3x3 matrices. a and newA are 3-vectors
__device__
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
__device__
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
__device__
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
__device__
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

/* Gradient of the isq function. */
// vec and grad are 3-vectors
__device__
void isqGradient(double *vec, double *grad, double e, double n)
{
    double xval = 0.0, yval = 0.0, zval = 0.0;
    // Check for n = 0 to prevent divide-by-zero errors
    if (n == 0)
    {
        xval = yval = zval = FLT_MAX;
    }
    // Check for e = 0 to prevent divide-by-zero errors
    else if (e == 0)
    {
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
__device__
double gPrime(double *vec, double *a, double e, double n)
{
    double tmp[3];
    isqGradient(vec, &tmp[0], e, n);
    double val = dot(a, &tmp[0]);
    delete[] tmp;
    return val;
}

/* Uses Newton's method to find the t value at which a ray hits the superquadric.
 * If the ray actually misses the superquadric then FLT_MAX is returned instead.*/
// a and b are 3-vectors
__device__
double updateRule(double *a, double *b, double e, double n, double t, double epsilon)
{
    double vec[3];
    findRay(a, b, &vec[0], t);
    double gP = gPrime(&vec[0], a, e, n);
    double gPPrevious = gP;
    double g = 0.0;
    double tnew = t, told = t;
    bool stopPoint = false;

    while (!stopPoint)
    {
        told = tnew;
        findRay(a, b, &vec[0], told);
        gP = gPrime(&vec[0], a, e, n);
        g = isq(&vec[0], e, n);

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



/* Unit normal vector at a point on the superquadric */
// r is a 3x3 matrix
// vec1, vec2, and un are 3-vectors
__device__
void unitNormal(double *r, double *vec1, double *vec2, double *un, double tt, double e, double n)
{
    findRay(vec1, vec2, un, tt);
    isqGradient(un, un, e, n);
    
    double un0 = un[0];
    double un1 = un[1];
    double un2 = un[2];
    
    un[0] = (r[0] * un0) + (r[1] * un1) + (r[2] * un2);
    un[1] = (r[3] * un0) + (r[4] * un1) + (r[5] * un2);
    un[2] = (r[6] * un0) + (r[7] * un1) + (r[8] * un2);
    
    normalize(un);
}

// Returns the angle between two vectors.
// Both a and b are 3-vectors.
__device__
double vectorAngle(double *a, double *b)
{
    double d = dot(a, b);
    double mag = norm(a) * norm(b);

    return acos(d / (double) mag);
}

// Calculates the refracted ray from an input ray and normal and a snell ratio
// If there is total internal reflection, then a vector of FLT_MAX is returned
// instead.
// a, n, and ref are 3-vectors
__device__
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

/********** Actual Raytracing Functions ***************************************/
__device__
// n is the normal. e is the eye. ind is the index of the object we're lighting.
void lighting(double *point, double *n, double *e,
              double *dif, double *amb, double *spec, double shine, 
              Point_Light *l, int numLights, 
              Object *objects, int numObjects,
              double epsilon, 
              int ind, int generation, double *res)
{
    double diffuseSum[] = {0.0, 0.0, 0.0};
    double specularSum[] = {0.0, 0.0, 0.0};
    double reflectedLight[] = {0.0, 0.0, 0.0};
    double refractedLight[] = {0.0, 0.0, 0.0};
    
    double newA[3];
    double newB[3];
    double coeffs[3];
    double roots[2];

    // Get the unit direction from the point to the camera
    double eDirection[3];
    for (int i = 0; i < 3; i++)
        eDirection[i] = e[i] - point[i];
        
    normalize(&eDirection[0]);

    for (int i = 0; i < numLights && generation > 0; i++)
    {
        // Retrieve the light's postion, color, and attenuation factor
        //Vector3d lP = l[i].position;
        //Vector3d lC = l[i].color;
        double attenuation = l[i].attenuation_k;

        // Get the unit direction and the distance between the light and the
        // point
        double lDirection[3];
        for (int j = 0; j < 3; j++)
            lDirection[i] = l[i].position[j] - point[i];
            
        double lightDist = norm(&lDirection[0]);
        normalize(&lDirection[0]);

        // Check to see that the light isn't blocked before considering it 
        // further. 
        // The i > 0 condition is present to prevent the program from blocking
        // anything from the eyelight, for the obvious reason that anything we
        // can see will be illuminated by the eyelight.
        bool useLight = true;
        for (int k = 0; k < numObjects && useLight && i > 0; k++)
        {
            if (k != ind)
            {
                // Find the ray equation transformations
                newa(objects[k].unScale, objects[k].unRotate, 
                     &lDirection[0], &newA[0]);
                newb(objects[k].unScale, objects[k].unRotate, 
                     objects[k].unTranslate, point, &newB[0]);

                // Find the quadratic equation coefficients
                findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                // Using the coefficients, find the roots
                findRoots(&coeffs[0], &roots[0]);

                // Check to see if the roots are FLT_MAX - if they are then the 
                // ray missed the superquadric. If they haven't missed then we 
                // can continue with the calculations.
                if (roots[0] != FLT_MAX)
                {
                    // Use the update rule to find tfinal
                    double tini = min(roots[0], roots[1]);
                    double tfinal = updateRule(&newA[0], &newB[0], objects[k].e, 
                                               objects[k].n, tini, epsilon);

                    /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                     * missed the superquadric. Additionally, if tfinal is 
                     * negative then either the ray has started inside the 
                     * object or is pointing away from the object; in both cases
                     * the ray has "missed". Also check to see if the object is
                     * farther away than the light - if it is then it isn't 
                     * actually blocking the light. */
                    double ray[3];
                    findRay(&lDirection[0], point, &ray[0], tfinal);
                    double objDist = norm(&ray[0]);
                    if (tfinal != FLT_MAX && tfinal >= 0 && objDist < lightDist)
                        useLight = false;
                }
            }
        }

        if (useLight)
        {
        
            // Find tthe attenuation term
            double atten = 1 / (double) (1 + (attenuation * pow(lightDist, 2)));
            // Add the attenuation factor to the light's color

            // Add the diffuse factor to the diffuse sum
            double nDotl = dot(n, &lDirection[0]);
            //Vector3d lDiffuse = lC * atten * ((0 < nDotl) ? nDotl : 0);
            //diffuseSum = diffuseSum + lDiffuse;
            if (0 < nDotl) {
                diffuseSum[0] += l[i].color[0] * atten * nDotl;
                diffuseSum[1] += l[i].color[1] * atten * nDotl;
                diffuseSum[2] += l[i].color[2] * atten * nDotl;
            }

            // Add the specular factor to the specular sum
            double dirDif[3];
            dirDif[0] = eDirection[0] + lDirection[0];
            dirDif[1] = eDirection[1] + lDirection[1];
            dirDif[2] = eDirection[2] + lDirection[2];
            normalize(&dirDif[0]);
            double nDotDir = dot(n, &dirDif[0]);
            //Vector3d lSpecular = lC * atten * 
            //             pow(((0 < nDotDir && 0 < nDotl) ? nDotDir : 0), shine);
            //specularSum = specularSum + lSpecular;
            if (0 < nDotDir && 0 < nDotl) {
                specularSum[0] += l[i].color[0] * atten * pow(nDotDir, shine);
                specularSum[1] += l[i].color[1] * atten * pow(nDotDir, shine);
                specularSum[2] += l[i].color[2] * atten * pow(nDotDir, shine);
            }
        }
    }
    /* Find the light contribution from reflection */
    // Find the reflected ray
    double eDotN = dot(n, &eDirection[0]);
    double reflected[3];
    reflected[0] = (2 * n[0] * eDotN) - eDirection[0];
    reflected[1] = (2 * n[1] * eDotN) - eDirection[1];
    reflected[2] = (2 * n[2] * eDotN) - eDirection[2];
    normalize(&reflected[0]);
    double ttrueFinal = 0.0;
    int finalObj = 0;
    double finalNewA[3];
    double finalNewB[3];
    bool hitObject = false;
    for (int k = 0; k < numObjects && generation > 0 ; k++)
    {
        if (k != ind)
        {
            // Find the ray equation transformations
            newa(objects[k].unScale, objects[k].unRotate, &reflected[0], 
                 &newA[0]);
            newb(objects[k].unScale, objects[k].unRotate, 
                 objects[k].unTranslate, point, &newB[0]);

            // Find the quadratic equation coefficients
            findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
            // Using the coefficients, find the roots
            findRoots(&coeffs[0], &roots[0]);

            // Check to see if the roots are FLT_MAX - if they are then the 
            // ray missed the superquadric. If they haven't missed then we 
            // can continue with the calculations.
            if (roots[0] != FLT_MAX)
            {
                // Use the update rule to find tfinal
                double tini = min(roots[0], roots[1]);
                double tfinal = updateRule(&newA[0], &newB[0], objects[k].e, 
                                           objects[k].n, tini, epsilon);

                /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                 * missed the superquadric. Additionally, if tfinal is negative 
                 * then either the ray has started inside the object or is 
                 * pointing away from the object; in both cases the ray has 
                 * "missed". */
                if (tfinal != FLT_MAX && tfinal >= 0)
                {
                    if(hitObject && tfinal < ttrueFinal)
                    {
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA[0] = newA[0];
                        finalNewA[1] = newA[1];
                        finalNewA[2] = newA[2];
                        finalNewB[0] = newB[0];
                        finalNewB[1] = newB[1];
                        finalNewB[2] = newB[2];
                    }
                    else if (!hitObject)
                    {
                        hitObject = true;
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA[0] = newA[0];
                        finalNewA[1] = newA[1];
                        finalNewA[2] = newA[2];
                        finalNewB[0] = newB[0];
                        finalNewB[1] = newB[1];
                        finalNewB[2] = newB[2];
                    }
                }
            }
        }
    }
    if (hitObject)
    {
        double intersectR[3];
        double intersectRNormal[3];
        findRay(&reflected[0], point, &intersectR[0], ttrueFinal);
        unitNormal(objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                   &intersectRNormal[0], ttrueFinal, objects[finalObj].e,
                   objects[finalObj].n);
                   
        lighting(&intersectR[0], &intersectRNormal[0], e,
                 objects[finalObj].mat->diffuse, 
                 objects[finalObj].mat->ambient, 
                 objects[finalObj].mat->specular, 
                 objects[finalObj].mat->shine, 
                 l, numLights, objects, numObjects, epsilon,
                 finalObj, generation-1, &reflectedLight[0]);
        if (shine < 1) {
            reflectedLight[0] *= shine;
            reflectedLight[1] *= shine;
            reflectedLight[2] *= shine;
        }
    }
    

    /* Find the refraction contribution. */
    // Change the eye-direction vector so that it points at the surface instead
    // of at the eye
    eDirection[0] *= -1;
    eDirection[1] *= -1;
    eDirection[2] *= -1;
    // Find the refracted ray
    double refracted1[3];
    refractedRay(&eDirection[0], n, &refracted1[0], objects[ind].mat->snell);
    normalize(&refracted1[0]);

    ttrueFinal = 0.0;
    finalObj = 0;
    hitObject = false;
    for (int k = 0; k < numObjects && generation > 0; k++)
    {
        if (k != ind)
        {
            // Find the ray equation transformations
            newa(objects[k].unScale, objects[k].unRotate, &refracted1[0], &newA[0]);
            newb(objects[k].unScale, objects[k].unRotate, 
                 objects[k].unTranslate, point, &newB[0]);

            // Find the quadratic equation coefficients
            findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
            // Using the coefficients, find the roots
            findRoots(&coeffs[0], &roots[0]);

            // Check to see if the roots are FLT_MAX - if they are then the 
            // ray missed the superquadric. If they haven't missed then we 
            // can continue with the calculations.
            if (roots[0] != FLT_MAX)
            {
                // Use the update rule to find tfinal
                double tini = min(roots[0], roots[1]);
                double tfinal = updateRule(&newA[0], &newB[0], objects[k].e, 
                                           objects[k].n, tini, epsilon);

                /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                 * missed the superquadric. Additionally, if tfinal is negative 
                 * then either the ray has started inside the object or is 
                 * pointing away from the object; in both cases the ray has 
                 * "missed". */
                if (tfinal != FLT_MAX && tfinal >= 0)
                {
                    if(hitObject && tfinal < ttrueFinal)
                    {
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA[0] = newA[0];
                        finalNewA[1] = newA[1];
                        finalNewA[2] = newA[2];
                        finalNewB[0] = newB[0];
                        finalNewB[1] = newB[1];
                        finalNewB[2] = newB[2];
                    }
                    else if (!hitObject)
                    {
                        hitObject = true;
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA[0] = newA[0];
                        finalNewA[1] = newA[1];
                        finalNewA[2] = newA[2];
                        finalNewB[0] = newB[0];
                        finalNewB[1] = newB[1];
                        finalNewB[2] = newB[2];
                    }
                }
            }
        }
    }
    if (hitObject)
    {
        double intersectR[3];
        double intersectRNormal[3];
        findRay(&refracted1[0], point, &intersectR[0], ttrueFinal);
        unitNormal(objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                   &intersectRNormal[0], ttrueFinal, objects[finalObj].e,
                   objects[finalObj].n);

        lighting(&intersectR[0], &intersectRNormal[0], e,
                 objects[finalObj].mat->diffuse, 
                 objects[finalObj].mat->ambient, 
                 objects[finalObj].mat->specular, 
                 objects[finalObj].mat->shine, 
                 l, numLights, objects, numObjects, epsilon,
                 finalObj, generation-1, &refractedLight[0]);
        refractedLight[0] *= objects[ind].mat->opacity;
        refractedLight[1] *= objects[ind].mat->opacity;
        refractedLight[2] *= objects[ind].mat->opacity;
    }
    else
    {
        double refA[3];
        double refB[3];
        double refCoeffs[3];
        double refRoots[3];
        newa(objects[ind].unScale, objects[ind].unRotate, &refracted1[0], &refA[0]);
        newb(objects[ind].unScale, objects[ind].unRotate, 
             objects[ind].unTranslate, point, &refB[0]);
        findCoeffs(&refA[0], &refB[0], &refCoeffs[0], true);
        findRoots(&refCoeffs[0], &refRoots[0]);

        double tini = max(refRoots[0], refRoots[1]);

        double tfinalRef = updateRule(&refA[0], &refB[0], objects[ind].e, 
                                      objects[ind].n, tini, epsilon);

        bool isRefracted = true;
        double outPoint[3];
        double outNormal[3];
        double outRay[3];
        if (isRefracted) // the fuck is the point of this?
        {
            findRay(&refracted1[0], point, &outPoint[0], tfinalRef);
            unitNormal(objects[ind].rotate, &refA[0], &refB[0], &outNormal[0], tfinalRef,
                       objects[ind].e, objects[ind].n);
            refractedRay(&refracted1[0], &outNormal[0], &outRay[0],
                         (double) 1 / objects[ind].mat->snell);
            // If the point has total internal reflection, then don't bother
            // with the rest of the refraction calculations.
            if(outRay[0] == FLT_MAX)
                isRefracted = false;
        }
        // Now that we've found where the ray exits, check to see if it hits any
        // objects; if it does, find the color contribution from that object
        ttrueFinal = 0.0;
        finalObj = 0;
        hitObject = false;
        for (int k = 0; k < numObjects && generation > 0 && isRefracted; k++)
        {
            if (k != ind)
            {
                // Find the ray equation transformations
                newa(objects[k].unScale, objects[k].unRotate, 
                     &outRay[0], &newA[0]);
                newb(objects[k].unScale, objects[k].unRotate, 
                     objects[k].unTranslate, &outPoint[0], &newB[0]);

                // Find the quadratic equation coefficients
                findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                // Using the coefficients, find the roots
                findRoots(&coeffs[0], &roots[0]);

                // Check to see if the roots are FLT_MAX - if they are then the 
                // ray missed the superquadric. If they haven't missed then we 
                // can continue with the calculations.
                if (roots[0] != FLT_MAX)
                {
                    // Use the update rule to find tfinal
                    double tini = min(roots[0], roots[1]);
                    double tfinal = updateRule(&newA[0], &newB[0], objects[k].e, 
                                               objects[k].n, tini, epsilon);

                    /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                     * missed the superquadric. Additionally, if tfinal is negative 
                     * then either the ray has started inside the object or is 
                     * pointing away from the object; in both cases the ray has 
                     * "missed". */
                    if (tfinal != FLT_MAX && tfinal >= 0)
                    {
                        if(hitObject && tfinal < ttrueFinal)
                        {
                            ttrueFinal = tfinal;
                            finalObj = k;
                            finalNewA[0] = newA[0];
                            finalNewA[1] = newA[1];
                            finalNewA[2] = newA[2];
                            finalNewB[0] = newB[0];
                            finalNewB[1] = newB[1];
                            finalNewB[2] = newB[2];
                        }
                        else if (!hitObject)
                        {
                            hitObject = true;
                            ttrueFinal = tfinal;
                            finalObj = k;
                            finalNewA[0] = newA[0];
                            finalNewA[1] = newA[1];
                            finalNewA[2] = newA[2];
                            finalNewB[0] = newB[0];
                            finalNewB[1] = newB[1];
                            finalNewB[2] = newB[2];
                        }
                    }
                }
            }
        }
        if (hitObject)
        {
            double intersectR[3];
            double intersectRNormal[3];
            findRay(&outRay[0], &outPoint[0], &intersectR[0], ttrueFinal);
            unitNormal(objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                       &intersectRNormal[0], ttrueFinal, objects[finalObj].e,
                       objects[finalObj].n);

            lighting(&intersectR[0], &intersectRNormal[0], e,
                     objects[finalObj].mat->diffuse, 
                     objects[finalObj].mat->ambient, 
                     objects[finalObj].mat->specular, 
                     objects[finalObj].mat->shine, 
                     l, numLights, objects, numObjects, epsilon,
                     finalObj, generation - 1, &refractedLight[0]);
            refractedLight[0] *= objects[ind].mat->opacity;
            refractedLight[1] *= objects[ind].mat->opacity;
            refractedLight[2] *= objects[ind].mat->opacity;
        }
    }

    double minVec[] = {1, 1, 1};
    double difProd[3];
    double specProd[3];
    double maxVec[3];
    cProduct(&diffuseSum[0], dif, &difProd[0]);
    cProduct(&specularSum[0], spec, &specProd[0]);
    maxVec[0] = difProd[0] + specProd[0] + reflectedLight[0] + refractedLight[0];
    maxVec[1] = difProd[1] + specProd[1] + reflectedLight[1] + refractedLight[1];
    maxVec[2] = difProd[2] + specProd[2] + reflectedLight[2] + refractedLight[2];
    cWiseMin(&minVec[0], &maxVec[0], &res[0]);
}

__global__
void raytraceKernel(Pixel *grid, Object *objects, double numObjects,
                    Point_Light *lightsPPM, double numLights, 
                    double Nx, double Ny, double filmX, double filmY, 
                    double *bgColor, double *e1, double *e2, double *e3, 
                    double *lookFrom, double epsilon, double filmDepth,
                    bool antiAliased)
{    
    double dx = filmX / (double) Nx;
    double dy = filmY / (double) Ny;

    double ttrueFinal = 0.0;
    int finalObj = 0;
    bool hitObject = false;
    
    double finalNewA[3];
    double finalNewB[3];
    double pointA[3];
    double newA[3];
    double newB[3];
    double coeffs[3];
    double roots[2];
    double intersect[3];
    double intersectNormal[3];
    
    // Parallize by screen pixel
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    while (i < Nx)
    {
        j = threadIdx.y + blockDim.y * blockIdx.y;
        
        while (j < Ny)
        {
            // The positions are subtracted by a Nx/2 or Ny/2 term to center
            // the film plane
            double px = (i * dx) - (filmX / (double) 2);
            double py = (j * dy) - (filmY / (double) 2);
            double pxColor[] = {bgColor[0], bgColor[1], bgColor[2]};
            if (!antiAliased)
            {
                findFilmA(px, py, e1, e2, e3, filmDepth, &pointA[0]);
                hitObject = false;
                finalObj = 0, ttrueFinal = 0;
                for (int k = 0; k < numObjects; k++)
                {
                    // Find the ray equation transformations
                    newa(objects[k].unScale, objects[k].unRotate, &pointA[0], &newA[0]);
                    newb(objects[k].unScale, objects[k].unRotate, 
                         objects[k].unTranslate, lookFrom, &newB[0]);

                    // Find the quadratic equation coefficients
                    findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                    // Using the coefficients, find the roots
                    findRoots(&coeffs[0], &roots[0]);

                    // Check to see if the roots are FLT_MAX - if they are then the 
                    // ray missed the superquadric. If they haven't missed then we 
                    // can continue with the calculations.
                    if (roots[0] != FLT_MAX)
                    {
                        // Use the update rule to find tfinal
                        double tini = min(roots[0], roots[1]);
                        double tfinal = updateRule(&newA[0], &newB[0], objects[k].e, 
                                                   objects[k].n, tini, epsilon);

                        /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                         * missed the superquadric. Additionally, if tfinal is negative 
                         * then either the ray has started inside the object or is 
                         * pointing away from the object; in both cases the ray has 
                         * "missed". */
                        if (tfinal != FLT_MAX && tfinal >= 0)
                        {
                            if(hitObject && tfinal < ttrueFinal)
                            {
                                ttrueFinal = tfinal;
                                finalObj = k;
                                finalNewA[0] = newA[0];
                                finalNewA[1] = newA[1];
                                finalNewA[2] = newA[2];
                                finalNewB[0] = newB[0];
                                finalNewB[1] = newB[1];
                                finalNewB[2] = newB[2];
                            }
                            else if (!hitObject)
                            {
                                hitObject = true;
                                ttrueFinal = tfinal;
                                finalObj = k;
                                finalNewA[0] = newA[0];
                                finalNewA[1] = newA[1];
                                finalNewA[2] = newA[2];
                                finalNewB[0] = newB[0];
                                finalNewB[1] = newB[1];
                                finalNewB[2] = newB[2];
                            }
                        }
                    }
                }
                if(hitObject)
                {
                    findRay(&pointA[0], lookFrom, &intersect[0], ttrueFinal);
                    unitNormal(objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                               &intersectNormal[0], ttrueFinal, objects[finalObj].e, 
                               objects[finalObj].n);

                    lighting(&intersect[0], &intersectNormal[0], lookFrom,
                             objects[finalObj].mat->diffuse, 
                             objects[finalObj].mat->ambient, 
                             objects[finalObj].mat->specular, 
                             objects[finalObj].mat->shine,
                             lightsPPM, numLights, objects, numObjects, epsilon,
                             finalObj, 3, &pxColor[0]);
                }
            }
            else
            {
                double denom = 3 + (2 / sqrt((double) 2));
                double pxCoeffs[] = {(1 / (2 * sqrt((double) 2))) / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt((double) 2))) / denom,
                                     (1 / (double) 2) / denom,
                                     1 / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt((double) 2))) / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt((double) 2))) / denom};
                int counter = 0;
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        double thisPx = px + (i * (dx / (double) 2));
                        double thisPy = py + (j * (dy / (double) 2));
                        findFilmA(thisPx, thisPy, e1, e2, e3, filmDepth, &pointA[0]);
                        hitObject = false;
                        finalObj = 0, ttrueFinal = 0;
                        for (int k = 0; k < numObjects; k++)
                        {
                            // Find the ray equation transformations
                            newa(objects[k].unScale, objects[k].unRotate, 
                                 &pointA[0], &newA[0]);
                            newb(objects[k].unScale, objects[k].unRotate, 
                                 objects[k].unTranslate, lookFrom, &newB[0]);

                            // Find the quadratic equation coefficients
                            findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                            // Using the coefficients, find the roots
                            findRoots(&coeffs[0], &roots[0]);

                            // Check to see if the roots are FLT_MAX - if they are then the 
                            // ray missed the superquadric. If they haven't missed then we 
                            // can continue with the calculations.
                            if (roots[0] != FLT_MAX)
                            {
                                // Use the update rule to find tfinal
                                double tini = min(roots[0], roots[1]);
                                double tfinal = updateRule(&newA[0], &newB[0], objects[k].e, 
                                                           objects[k].n, tini, epsilon);

                                /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                                 * missed the superquadric. Additionally, if tfinal is negative 
                                 * then either the ray has started inside the object or is 
                                 * pointing away from the object; in both cases the ray has 
                                 * "missed". */
                                if (tfinal != FLT_MAX && tfinal >= 0)
                                {
                                    if(hitObject && tfinal < ttrueFinal)
                                    {
                                        ttrueFinal = tfinal;
                                        finalObj = k;
                                        finalNewA[0] = newA[0];
                                        finalNewA[1] = newA[1];
                                        finalNewA[2] = newA[2];
                                        finalNewB[0] = newB[0];
                                        finalNewB[1] = newB[1];
                                        finalNewB[2] = newB[2];
                                    }
                                    else if (!hitObject)
                                    {
                                        hitObject = true;
                                        ttrueFinal = tfinal;
                                        finalObj = k;
                                        finalNewA[0] = newA[0];
                                        finalNewA[1] = newA[1];
                                        finalNewA[2] = newA[2];
                                        finalNewB[0] = newB[0];
                                        finalNewB[1] = newB[1];
                                        finalNewB[2] = newB[2];
                                    }
                                }
                            }
                        }
                        if(hitObject)
                        {
                            double intersect[3];
                            double intersectNormal[3];
                            findRay(&pointA[0], lookFrom, &intersect[0], ttrueFinal);
                            unitNormal(objects[finalObj].rotate, &finalNewA[0], 
                                       &finalNewB[0], &intersectNormal[0], ttrueFinal, 
                                       objects[finalObj].e, objects[finalObj].n);

                            double color[] = {0, 0, 0};
                            
                            lighting(&intersect[0], &intersectNormal[0], lookFrom,
                                     objects[finalObj].mat->diffuse, 
                                     objects[finalObj].mat->ambient, 
                                     objects[finalObj].mat->specular, 
                                     objects[finalObj].mat->shine,
                                     lightsPPM, numLights, objects, numObjects, 
                                     epsilon,
                                     finalObj, 3, &color[0]);

                            pxColor[0] += color[0] * pxCoeffs[counter];
                            pxColor[1] += color[1] * pxCoeffs[counter];
                            pxColor[2] += color[2] * pxCoeffs[counter];
                        }
                        counter++;
                    }
                }
                
            }
            int index = j * (int) Ny + i;
            grid[index].red = pxColor[0];
            grid[index].green = pxColor[1];
            grid[index].blue = pxColor[2];
            
            
            j += blockDim.y * gridDim.y;
        }
        i += blockDim.x * gridDim.x;
    }
    
    // can you use delete[] in cuda...?
    delete[] finalNewA;
    delete[] finalNewB;
    delete[] pointA;
    delete[] newA;
    delete[] newB;
    delete[] coeffs;
    delete[] roots;
    delete[] intersect;
    delete[] intersectNormal;
}

void callRaytraceKernel(Pixel *grid, Object *objs, double numObjects,
                        Point_Light *lightsPPM, double numLights, double Nx, 
                        double Ny, double filmX, double filmY, 
                        double *bgColor, double *e1, double *e2, double *e3, 
                        double *lookFrom, double epsilon, double filmDepth,
                        bool antiAliased, int blockPower) 
{
    int blockSize = pow(2, blockPower);
    
    // about 1 thread per screen pixel
    dim3 blocks(blockSize, blockSize);
    int gx = Nx / blockSize;
    int gy = Ny / blockSize;
    if (gx < 1) gx = 1;
    if (gy < 1) gy = 1;
    dim3 grids(Nx / blockSize, Ny / blockSize);
    
    raytraceKernel<<<grids, blocks>>>(grid, objs, numObjects, lightsPPM,
                                      numLights, Nx, Ny, filmX, filmY, bgColor,
                                      e1, e2, e3, lookFrom, epsilon, filmDepth,
                                      antiAliased);
}
