#include <cassert>
#include <cuda_runtime.h>
#include <math.h>
#include "raytraceCuda.cuh"
#include "util.h"

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
    double *scale;      //3x3-matrix
    double *unScale;    //3x3-matrix
    double *rotate;     //3x3-matrix
    double *unRotate;   //3x3-matrix
    double *translate;  //3-vector
    double *unTranslate; //3-vector
};

struct Pixel
{
    double red;
    double green;
    double blue;
};
               
// Function declarations
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

__device__
// n is the normal. e is the eye. ind is the index of the object we're lighting.
void lighting(double *point, double *n, double *e,
              double *dif, double *amb, double *spec, double shine, 
              Point_Light *l, int numLights, 
              Object *objects, int numObjects, 
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
                newb(&objects[k].unScale, &objects[k].unRotate, 
                     &objects[k].unTranslate, point, &newB[0]);

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
                    double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                               &objects[k].n, tini, epsilon);

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
            newa(&objects[k].unScale, &objects[k].unRotate, &reflected[0], 
                 &newA[0]);
            newb(&objects[k].unScale, &objects[k].unRotate, 
                 &objects[k].unTranslate, point, &newB[0]);

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
                double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                           &objects[k].n, tini, epsilon);

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
        unitNormal(&objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                   &intersectRNormal[0], ttrueFinal, &objects[finalObj].e,
                   &objects[finalObj].n);
                   
        lighting(&intersectR[0], &intersectRNormal[0], e,
                 &objects[finalObj].mat.diffuse, 
                 &objects[finalObj].mat.ambient, 
                 &objects[finalObj].mat.specular, 
                 &objects[finalObj].mat.shine, 
                 l, numLights, objects, numObjects,
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
    refractedRay(&eDirection[0], n, &refracted1[0], &objects[ind].mat.snell);
    normalize(&refracted1[0]);

    ttrueFinal = 0.0;
    finalObj = 0;
    hitObject = false;
    for (int k = 0; k < numObjects && generation > 0; k++)
    {
        if (k != ind)
        {
            // Find the ray equation transformations
            newa(&objects[k].unScale, &objects[k].unRotate, &refracted1[0], &newA[0]);
            newb(&objects[k].unScale, &objects[k].unRotate, 
                 &objects[k].unTranslate, point, &newB[0]);

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
                double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                           &objects[k].n, tini, epsilon);

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
        findRay(&refracted1[0], point, &intersect[0], ttrueFinal);
        unitNormal(&objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                   &intersectRNormal[0], ttrueFinal, &objects[finalObj].e,
                   &objects[finalObj].n);

        lighting(&intersectR[0], &intersectRNormal[0], e,
                 &objects[finalObj].mat.diffuse, 
                 &objects[finalObj].mat.ambient, 
                 &objects[finalObj].mat.specular, 
                 &objects[finalObj].mat.shine, 
                 l, numLights, objects, numObjects,
                 finalObj, generation-1, &refractedLight[0]);
        refractedLight[0] *= objects[ind].mat.opacity;
        refractedLight[1] *= objects[ind].mat.opacity;
        refractedLight[2] *= objects[ind].mat.opacity;
    }
    else
    {
        double refA[3];
        double refB[3];
        double refCoeffs[3];
        double refRoots[3];
        newa(&objects[ind].unScale, &objects[ind].unRotate, &refracted1[0], &refA[0]);
        newb(&objects[ind].unScale, &objects[ind].unRotate, 
             &objects[ind].unTranslate, point, &refB[0]);
        findCoeffs(&refA[0], &refB[0], &refCoeffs[0], true);
        findRoots(&refCoeffs[0], &refRoots[0]);

        double tini = max(refRoots(0), refRoots(1));

        double tfinalRef = updateRule(&refA[0], &refB[0], &objects[ind].e, 
                                      &objects[ind].n, tini, epsilon);

        bool isRefracted = true;
        double outPoint[3];
        double outNormal[3];
        double outRay[3];
        if (isRefracted) // the fuck is the point of this?
        {
            findRay(&refracted1[0], point, &outPoint[0], tfinalRef);
            unitNormal(&objects[ind].rotate, &refA[0], &refB[0], &outNormal[0], tfinalRef,
                       &objects[ind].e, &objects[ind].n);
            refractedRay(&refracted1[0], &outNormal[0], &outRay[0],
                         (double) 1 / objects[ind].mat.snell);
            // If the point has total internal reflection, then don't bother
            // with the rest of the refraction calculations.
            if(outRay(0) == FLT_MAX)
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
                newa(&objects[k].unScale, &objects[k].unRotate, 
                     &outRay[0], &newA[0]);
                newb(&objects[k].unScale, &objects[k].unRotate, 
                     &objects[k].unTranslate, &outPoint[0], &newB[0]);

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
                    double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                               &objects[k].n, tini, epsilon);

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
            unitNormal(&objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                       &intersectRNormal[0], ttrueFinal, &objects[finalObj].e,
                       &objects[finalObj].n);

            lighting(&intersectR[0], &intersectRNormal[0], e,
                     &objects[finalObj].mat.diffuse, 
                     &objects[finalObj].mat.ambient, 
                     &objects[finalObj].mat.specular, 
                     &objects[finalObj].mat.shine, 
                     l, numLights, objects, numObjects,
                     finalObj, generation - 1, &refractedLight[0]);
            refractedLight[0] *= objects[ind].mat.opacity;
            refractedLight[1] *= objects[ind].mat.opacity;
            refractedLight[2] *= objects[ind].mat.opacity;
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
            if (!antiAlias)
            {
                findFilmA(px, py, e1, e2, e3, &pointA[0]);
                hitObject = false;
                finalObj = 0, ttrueFinal = 0;
                for (int k = 0; k < numObjects; k++)
                {
                    // Find the ray equation transformations
                    newa(&objects[k].unScale, &objects[k].unRotate, &pointA[0], &newA[0]);
                    newb(&objects[k].unScale, &objects[k].unRotate, 
                         &objects[k].unTranslate, lookFrom, &newB[0]);

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
                        double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                                   &objects[k].n, tini, epsilon);

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
                    unitNormal(&objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                               &intersectNormal[0], ttrueFinal, &objects[finalObj].e, 
                               &objects[finalObj].n);

                    lighting(&intersect[0], &intersectNormal[0], lookFrom,
                             &objects[finalObj].mat.diffuse, 
                             &objects[finalObj].mat.ambient, 
                             &objects[finalObj].mat.specular, 
                             &objects[finalObj].mat.shine, 
                             lightsPPM, numLights, objects, numObjects, 
                             finalObj, 3, &pxColor[0]);
                }
            }
            else
            {
                double denom = 3 + (2 / sqrt(2));
                double pxCoeffs[] = {(1 / (2 * sqrt(2))) / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt(2))) / denom,
                                     (1 / (double) 2) / denom,
                                     1 / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt(2))) / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt(2))) / denom};
                int counter = 0;
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        double thisPx = px + (i * (dx / (double) 2));
                        double thisPy = py + (j * (dy / (double) 2));
                        findFilmA(thisPx, thisPy, e1, e2, e3, &pointA[0]);
                        hitObject = false;
                        finalObj = 0, ttrueFinal = 0;
                        for (int k = 0; k < numObjects; k++)
                        {
                            // Find the ray equation transformations
                            newa(&objects[k].unScale, &objects[k].unRotate, 
                                 &pointA[0], &newA[0]);
                            newb(&objects[k].unScale, &objects[k].unRotate, 
                                 &objects[k].unTranslate, lookFrom, &newB[0]);

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
                                double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                                           &objects[k].n, tini, epsilon);

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
                            unitNormal(&objects[finalObj].rotate, &finalNewA[0], 
                                       &finalNewB[0], &intersectNormal[0], ttrueFinal, 
                                       &objects[finalObj].e, &objects[finalObj].n);

                            double color[] = {0, 0, 0};
                            
                            lighting(&intersect[0], &intersectNormal[0], lookFrom,
                                     &objects[finalObj].mat.diffuse, 
                                     &objects[finalObj].mat.ambient, 
                                     &objects[finalObj].mat.specular, 
                                     &objects[finalObj].mat.shine, 
                                     lightsPPM, numLights, objects, numObjects, 
                                     finalObj, 3, &color[0]);

                            pxColor[0] += color[0] * pxCoeffs[counter];
                            pxColor[1] += color[1] * pxCoeffs[counter];
                            pxColor[2] += color[2] * pxCoeffs[counter];
                        }
                        counter++;
                    }
                }
                
            }
            grid[j * Ny + i]->red = pxColor[0];
            grid[j * Ny + i]->green = pxColor[1];
            grid[j * Ny + i]->blue = pxColor[2];
            
            
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
