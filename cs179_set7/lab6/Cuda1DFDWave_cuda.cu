/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/* TODO: You'll need a kernel here, as well as any helper functions
to call it */

__global__
void waveEquationKernal(float *old_data, float *current_data, float *new_data,
                        int numberOfNodes, float constant) {
    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < numberOfNodes - 1) {
        // This is to make sure that thread index 0 can still move on to the
        // next thread at blockDim.x * gridDim.x
        if (index > 0) {
            // Wave Equation!
            // y_x,t+1 = 2*y_x,t - y_x,t-1 + 
            //                       (c*dt/dx)^2 * (y_x+1,t - 2*y_x,t + y_x-1,t)
            new_data[index] = 2 * current_data[index] 
                              - old_data[index]
                              + constant 
                              * (current_data[index + 1] 
                                 - 2 * current_data[index] 
                                 + current_data[index - 1]);
        }
        index += blockDim.x * gridDim.x;
    }
}

void waveEquation(float *old_data, float *current_data, float *new_data,
                  int numberOfNodes, float c, float dt, float dx,
                  int blocks, int threadsPerBlock) {

    float constant = ((c * c * dt * dt) / (dx * dx));
    waveEquationKernal<<<blocks, threadsPerBlock>>>(old_data, current_data,
                                                    new_data, numberOfNodes,
                                                    constant);
}
