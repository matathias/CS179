
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>


#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include <algorithm>

#include "GillespieCuda.cuh"

#define SimulationCount 1000
#define NumTimePoints   1000
#define NumSeconds      100

int main(int argc, char* argv[]) {
    
    if (argc < 3){
        printf("Usage: (threads per block) (max number of blocks)\n");
        exit(-1);
    }
    const unsigned int threadsPerBlock = atoi(argv[1]);
    const unsigned int blocks = atoi(argv[2]);
    
    
    /***** Allocate all the data~ *****/
    // Allocate the cpu's data
    float *expectations = (float*)malloc(NumTimePoints * sizeof(float));
    float *variance = (float*)malloc(NumTimePoints * sizeof(float));
    int *done = (float*)malloc(sizeof(int));
    *done = 0;
    
    // Allocate the gpu's data
    int *d_productionStates, *d_concentrations;
    int *d_oldConcentrations, *d_newConcentrations;
    float *d_times, *d_randomTimeSteps, *d_randomProbs;
    float *d_expectations, *d_variance;
    int *d_done;
    curandState_t *d_states;
    
    cudaMalloc(&d_productionStates, SimulationCount * sizeof(int));
    cudaMalloc(&d_concentrations, SimulationCount * NumTimePoints * sizeof(int));
    cudaMalloc(&d_oldConcentrations, SimulationCount * sizeof(int));
    cudaMalloc(&d_newConcentrations, SimulationCount * sizeof(int));
    cudaMalloc(&d_times, SimulationCount * sizeof(float));
    cudaMalloc(&d_randomTimeSteps, SimulationCount * sizeof(float));
    cudaMalloc(&d_randomProbs, SimulationCount * sizeof(float));
    cudaMalloc(&d_expectations, NumTimePoints * sizeof(float));
    cudaMalloc(&d_variance, NumTimePoints * sizeof(float));
    cudaMalloc(&d_done, sizeof(int));
    cudaMalloc(&d_states, 2 * sizeof(curandState_t));
    
    // Initialize everything to 0 that needs to be set to 0
    cudaMemset(&d_productionStates, 0, SimulationCount * sizeof(int));
    cudaMemset(&d_concentrations, 0, SimulationCount * NumTimePoints 
                                     * sizeof(int));
    cudaMemset(&d_oldConcentrations, 0, SimulationCount * sizeof(int));
    cudaMemset(&d_newConcentrations, 0, SimulationCount * sizeof(int));
    cudaMemset(&d_times, 0, SimulationCount * sizeof(int));
    
    
    /***** Start the simulations *****/
    // Loop from time 0 to time NumSeconds; we use a while loop because the
    // timesteps are variable. We will stop when the value of done is still 1
    // after resampleKernel is run.
    while(! *done) {
        *done = 1;
        cudaMemcpy(d_done, done, sizeof(int), cudaMemcpyHostToDevice);
        
        callGillespieKernel(d_productionStates, d_oldConcentrations,
                            d_newConcentrations, d_times, d_randomTimeSteps,
                            d_randomProbs, d_states, SimulationCount, blocks,
                            threadsPerBlock);
                            
        callResampleKernel(d_concentrations, d_newConcentrations, d_times,
                           SimulationCount, 
                           (float) NumTimePoints / (float) NumSeconds,
                           NumSeconds, d_done, blocks, threadsPerBlock);
        
        // Copy d_done into done so we can know whether to stop or continue
        cudaMemcpy(done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Find the expectation and variance of the concentrations
    callBehaviorKernel(d_concentrations, d_expectations, d_variance,
                       SimulationCount, NumTimePoints, blocks, threadsPerBlock);
    
    /***** Output the results *****/
    // Copy data back to the host
    cudaMemcpy(expectations, d_expectations, NumTimePoints * sizeof(float), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(variance, d_variance, NumTimePoints * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Print out the expectations and concentrations
    for (int i = 0; i < NumTimePoints; i++) {
        float timestamp = i * ((float) NumSeconds / (float) NumTimePoints);
        printf("TIME (s): %f\t Expectation: %f\t Variance: %f\n",
                timestamp, expectations[i], variance[i]);
    }
    
    // Free all the data
    free(expectations);
    free(variance);
    free(done);
    
    cudaFree(d_productionStates);
    cudaFree(d_concentations);
    cudaFree(d_oldConcentrations);
    cudaFree(d_newConcentrations);
    cudaFree(d_times);
    cudaFree(d_randomTimeSteps);
    cudaFree(d_randomProbs);
    cudaFree(d_expectations);
    cudaFree(d_variance);
    cudaFree(d_done);
    cudaFree(d_states);
}
