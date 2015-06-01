
#include <cstdio>
#include <math.h>

#include <cuda_runtime.h>

#include "util.h"

void callRaytraceKernel(Pixel *grid, Object *objs, double numObjects,
                        Point_Light *lightsPPM, double numLights, double Nx, 
                        double Ny, double filmX, double filmY, 
                        double *bgColor, double *e1, double *e2, double *e3, 
                        double *lookFrom, double epsilon, double filmDepth,
                        bool antiAliased, int blockPower);
