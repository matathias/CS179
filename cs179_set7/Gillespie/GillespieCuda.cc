
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>


#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <algorithm>

#include "GillespieCuda.cuh"

#define SimulationCount 1000
#define NumTimePoints   1000
#define NumSeconds      100

#define DEBUG 1

using std::cerr;
using std::cout;
using std::endl;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

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
    int *done = (int*)malloc(sizeof(int));
    memset(done, 0, sizeof(int));
    
    // Allocate the gpu's data
    int *d_productionStates, *d_concentrations;
    int *d_oldConcentrations, *d_newConcentrations;
    float *d_times, *d_randomTimeSteps, *d_randomProbs;
    float *d_expectations, *d_variance;
    int *d_done;
    curandState_t *d_states;
    
    gpuErrChk(cudaMalloc(&d_productionStates, SimulationCount * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_concentrations, 
                         SimulationCount * NumTimePoints * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_oldConcentrations, SimulationCount * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_newConcentrations, SimulationCount * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_times, SimulationCount * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_randomTimeSteps, SimulationCount * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_randomProbs, SimulationCount * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_expectations, NumTimePoints * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_variance, NumTimePoints * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_done, sizeof(int)));
    gpuErrChk(cudaMalloc(&d_states, 2 * sizeof(curandState_t)));
    
    // Initialize everything to 0 that needs to be set to 0
    gpuErrChk(cudaMemset(d_productionStates, 0, SimulationCount * sizeof(int)));
    gpuErrChk(cudaMemset(d_concentrations, 0, SimulationCount * NumTimePoints 
                                              * sizeof(int)));
    gpuErrChk(cudaMemset(d_oldConcentrations, 0, SimulationCount * sizeof(int)));
    gpuErrChk(cudaMemset(d_newConcentrations, 0, SimulationCount * sizeof(int)));
    gpuErrChk(cudaMemset(d_times, 0, SimulationCount * sizeof(int)));
    
    
    /***** Start the simulations *****/
    // Loop from time 0 to time NumSeconds; we use a while loop because the
    // timesteps are variable. We will stop when the value of done is still 1
    // after resampleKernel is run.
#if DEBUG
    printf("Done value: %d\n", done[0]);
    float *o_times = (float*)malloc(SimulationCount * sizeof(float));
    int *o_oldCon = (int*)malloc(SimulationCount * sizeof(float));
    int *o_newCon = (int*)malloc(SimulationCount * sizeof(float));
    float *o_randP = (float*)malloc(SimulationCount * sizeof(float));
    float *o_randT = (float*)malloc(SimulationCount * sizeof(float));
#endif
    while(done[0] == 0) {
        done[0] = 1;
        gpuErrChk(cudaMemcpy(d_done, done, sizeof(int), cudaMemcpyHostToDevice));
        
        time_t t;
        time(&t);
        
        curandGenerator_t gen_timeSteps;
        curandCreateGenerator(&gen_timeSteps, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen_timeSteps, (unsigned long) t);
        curandGenerateUniform(gen_timeSteps, d_randomTimeSteps, SimulationCount);
        
        curandGenerator_t gen_randomProbs;
        curandCreateGenerator(&gen_randomProbs, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen_randomProbs, ((unsigned long) t + 10));
        curandGenerateUniform(gen_randomProbs, d_randomProbs, SimulationCount);
        
        callGillespieKernel(d_productionStates, d_oldConcentrations,
                            d_newConcentrations, d_times, d_randomTimeSteps,
                            d_randomProbs, d_states, SimulationCount, blocks,
                            threadsPerBlock);
        
        
#if DEBUG
        // Let's see what's in the gpu...
        gpuErrChk(cudaMemcpy(o_times, d_times, SimulationCount * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(o_oldCon, d_oldConcentrations, SimulationCount * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(o_newCon, d_newConcentrations, SimulationCount * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(o_randP, d_randomProbs, SimulationCount * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(o_randT, d_randomTimeSteps, SimulationCount * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < SimulationCount; i+=100){
            printf("Time for simulation %d: %f\n", i, o_times[i]);
            printf("\tOld Concentration: %d\n", o_oldCon[i]);
            printf("\tNew Concentration: %d\n", o_newCon[i]);
            printf("\tRandom Prob:     %f\n", o_randP[i]);
            printf("\tRandom Time val: %f\n", o_randT[i]);
        }
#endif
        
        callResampleKernel(d_concentrations, d_newConcentrations, d_times,
                           SimulationCount, 
                           (float) NumTimePoints / (float) NumSeconds,
                           NumSeconds, d_done, blocks, threadsPerBlock);
        
        // Copy d_done into done so we can know whether to stop or continue
        gpuErrChk(cudaMemcpy(done, d_done, sizeof(int), cudaMemcpyDeviceToHost));
#if DEBUG
        printf("Done value (loop): %d\n", done[0]);
#endif
        
        // point d_oldConcentrations to d_newConcentrations and vice versa
        int *tmp = d_oldConcentrations;
        d_oldConcentrations = d_newConcentrations;
        d_newConcentrations = tmp;
    }
    
    // Find the expectation and variance of the concentrations
    callBehaviorKernel(d_concentrations, d_expectations, d_variance,
                       SimulationCount, NumTimePoints, blocks, threadsPerBlock);
    
    /***** Output the results *****/
    // Copy data back to the host
    gpuErrChk(cudaMemcpy(expectations, d_expectations, NumTimePoints * sizeof(float), 
               cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(variance, d_variance, NumTimePoints * sizeof(float), 
               cudaMemcpyDeviceToHost));
    
    // Print out the expectations and concentrations
#if DEBUG
    for (int i = NumTimePoints-20; i < NumTimePoints; i++) {
#else
    for (int i = 0; i < NumTimePoints; i++) {
#endif
        float timestamp = i * ((float) NumSeconds / (float) NumTimePoints);
        printf("TIME (s): %4.1f\t Expectation: %f\t Variance: %f\n",
                timestamp, expectations[i], variance[i]);
    }
    
    // Free all the data
    free(expectations);
    free(variance);
    free(done);
    
    gpuErrChk(cudaFree(d_productionStates));
    gpuErrChk(cudaFree(d_concentrations));
    gpuErrChk(cudaFree(d_oldConcentrations));
    gpuErrChk(cudaFree(d_newConcentrations));
    gpuErrChk(cudaFree(d_times));
    gpuErrChk(cudaFree(d_randomTimeSteps));
    gpuErrChk(cudaFree(d_randomProbs));
    gpuErrChk(cudaFree(d_expectations));
    gpuErrChk(cudaFree(d_variance));
    gpuErrChk(cudaFree(d_done));
    gpuErrChk(cudaFree(d_states));
}
