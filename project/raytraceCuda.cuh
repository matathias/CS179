
#include <cstdio>
#include <math.h>

#include <cuda_runtime.h>

struct Point_Light;
struct Material;
struct Object;
struct Pixel;

void callRaytraceKernel(double *grid, Object *objs, double numObjects,
                        Point_Light *lightsPPM, double numLights, int Nx, 
                        int Ny, double filmX, double filmY, 
                        double *bgColor, double *e1, double *e2, double *e3, 
                        double *lookFrom, double epsilon, double filmDepth,
                        bool antiAliased, int blockPower);
