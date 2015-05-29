/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/* TODO: You'll need a kernel here, as well as any helper functions
to call it */

//__global__
void waveEquationKernal(float *old_data, float *current_data, float *new_data,
                        int numberOfNodes, float c, float dt, float dx) {
    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index > 0 && index < numberOfNodes - 1) {
        // Wave Equation!
        // y_x,t+1 = 2*y_x,t - y_x,t-1 + 
        //                           (c*dt/dx)^2 * (y_x+1,t - 2*y_x,t + y_x-1,t)
        new_data[index] = 2 * current_data[index] - 
                          old_data[index] + 
                          ((c * c * dt * dt) / (dx * dx)) *
                          (current_data[index + 1] -
                           2 * current_data[index] +
                           current_data[index - 1]);
        
        index += blockDim.x * gridDim.x;
    }
}
