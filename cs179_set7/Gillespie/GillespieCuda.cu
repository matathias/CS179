
#include <cstdio>
#include <math.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#include "GillespieCuda.cuh"

#define b 10.0f
#define g 1.0f
#define Kon 0.1f
#define Koff 0.9f

/* This kernel performs a single iteration of the gillespie algorithm.
 *
 * productionStates is an array of 0s and 1s where each value signals whether or
 * not the simulation of the same index is ON (1) or OFF (0).
 * 
 * *_concentrations are arrays of concentrations, one for each simulation
 * 
 * times is an array containing the time of each simulation
 * 
 * Both "random" arrays are numSimulation-length 1D arrays containing random
 * values between 0 and 1. randomTimeSteps is used to calculate the next time
 * step for each simulation, and randomProbs is used to find which transition
 * each simulation undergoes.
 */
__global__
void singleGillespieKernel(int *productionStates, int *old_concentrations, 
                           int *new_concentrations, float *times, 
                           float *randomTimeSteps, float *randomProbs, 
                           int numSimulations) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (index < numSimulations) {
        float rate;
        if (productionStates[index]) { // Production is active
            rate = Koff + b + old_concentrations[index] * g;
            if (randomProbs[index] < Koff / rate) {
                // If the random number is less than Koff/rate
                productionStates[index] = 0; // production becomes inactive
                new_concentrations[index] = old_concentrations[index];
            }
            else if (randomProbs[index] < (Koff + b) / rate) {
                // If the random number is more than Koff/rate but less than
                // (Koff + b) / rate; i.e. the probability of the concentration
                // increasing
                new_concentrations[index] = old_concentrations[index] + 1;
            }
            else {
                // If the random number is more than (Koff + b) / rate, i.e. the
                // probability of the concentration decreasing
                new_concentrations[index] = old_concentrations[index] - 1;
            }
        }
        else { // Production is inactive
            rate = Kon + old_concentrations[index] * g;
            if (randomProbs[index] < Kon / rate) {
                // If the system becomes active...
                productionStates[index] = 1;
                new_concentrations[index] = old_concentrations[index];
            }
            else {
                // If the concentration decays...
                new_concentrations[index] = old_concentrations[index] - 1;
            }
        }
        
        // Update the time for the simulation
        times[index] += -1 * log(randomTimeSteps[index]) / rate;
        index += blockDim.x * gridDim.x;
    }
}

/* This kernel updates the actual production state and concentration arrays as
 * necessary for every timepoint.
 * 
 * concentrations is a 2-dimensional array that keeps track of the concentration
 * of every simulation at every timestep. 
 * The array is size (numSimulations x numTimeSteps).
 *
 * new_concentrations is a 1D array produced from the singleGillespieKernal. 
 * It is size numSimulations.
 * 
 * times is a 1D array of size numSimulations that contains the current time
 * for the simulation of the corresponding index
 *
 * numSimulations is the number of simulations.
 *
 * timeFactor is the factor we use to multiply a time (in seconds) to get an
 * index in the productionStates or concentrations array. For example, if we
 * want 1000 points covering 100 seconds, then timeFactor would be 
 * 1000 / 100 = 10.
 *
 * endTime is the time at which a simulation should be stopped.
 *
 * done is a boolean value signalling whether or not all simulations are done.
 * It is set to 1 before the call to the kernel; if any thread is still
 * running, then the thread will set done to 0, which can be read after the
 * completion of the kernel.
 */
__global__
void resampleKernel(int *concentrations, int *new_concentrations,
                    float *times, int numSimulations, float timeFactor,
                    float endTime, int *done) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (index < numSimulations) {
        // If the simulation's time is less than the end time, then continue.
        // Otherwise, skip updates for this simulation.
        if (times[index] < endTime) {
            // A thread is still executing, so we are NOT done
            done[index] = 1;
            
            // Get the timestep in seconds and transform this into an index for 
            // the concentrations array (which is 2D)
            int timeIndex = (int) (times[index] * timeFactor);
            
            concentrations[timeIndex * numSimulations + index] =
                new_concentrations[index];
        }
        
        index += blockDim.x * gridDim.x;
    }
}

/* This kernel calculates the expected concentration and variance at every
 * timepoints.
 *
 * concentrations is a 2D array containing the concentration of every simulation
 * at every timepoint.
 *
 * expectations and variance are both 1D arrays of length numTimes, containing
 * the expectation and variance (respectively) of the concentration at each
 * timepoint.
 *
 * numSimulations is the number of simulations.
 *
 * numTimes is the number of timepoints.
 */
__global__
void behaviorKernel(int *concentrations, float *expectations, 
                    float *variance, int numSimulations, int numTimes) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles every concentration at a single timepoint
    while (index < numTimes) {
        float avg = 0;
        float var = 0;
        for (int i = 0; i < numSimulations; i++) {
            float val = concentrations[index * numSimulations + i];
            avg += val;
            var += val * val;
        }
        
        // Get the expectation by dividing avg by the number of simulations
        avg = (float) avg / (float) numSimulations;
        
        // Get the variance
        var = ((float) var / (float) numSimulations) - (avg * avg);
        
        expectations[index] = avg;
        variance[index] = var;
    
        index += blockDim.x * gridDim.x;
    }
}

/* This kernel fills the array randoms with random numbers of a uniform
 * distribution between 0 and 1.
 */
__global__
void randomNumberKernel(curandState_t *state, float *randoms, float numRandoms){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (index < numRandoms) {
        randoms[index] = curand_uniform(state);
    
        index += blockDim.x * gridDim.x;
    }
}

/*
 * This kernel sets up the curand states for the random kernel above.
 */
__global__
void setupCurandKernel(curandState_t *state, unsigned long seed) {
    curand_init(seed, 0, 0, state);
}

void callGillespieKernel(int *productionStates, 
                         int *old_concentrations, int *new_concentrations,
                         float *times, float *randomTimeSteps,
                         float *randomProbs, curandState_t *state,
                         int numSimulations,
                         int blocks, int threadsPerBlock) {
    // Calculate a seed and initialize the states
    time_t t;
    time(&t);
    
    setupCurandKernel<<<blocks, threadsPerBlock>>>(&state[0], 
                                                   (unsigned long) t);
    setupCurandKernel<<<blocks, threadsPerBlock>>>(&state[1], 
                                                   ((unsigned long) t) + 10);
    
    // Fill randomTimeSteps and randomProbs with random values
    randomNumberKernel<<<blocks, threadsPerBlock>>>(&state[0], randomTimeSteps, 
                                                    numSimulations);
    randomNumberKernel<<<blocks, threadsPerBlock>>>(&state[1], randomProbs,
                                                    numSimulations);
    
    // Now call the Gillespie kernel
    singleGillespieKernel<<<blocks, threadsPerBlock>>>(productionStates,
                                                       old_concentrations,
                                                       new_concentrations,
                                                       times, randomTimeSteps,
                                                       randomProbs,
                                                       numSimulations);
}

void callResampleKernel(int *concentrations, int *new_concentrations,
                        float *times, int numSimulations, float timeFactor,
                        float endTime, int *done, int blocks, 
                        int threadsPerBlock) {
    resampleKernel<<<blocks, threadsPerBlock>>>(concentrations,
                                                new_concentrations,
                                                times, numSimulations,
                                                timeFactor, endTime, done);
}

void callBehaviorKernel(int *concentrations, float *expectations, 
                        float *variance, int numSimulations, int numTimes,
                        int blocks, int threadsPerBlock) {
    behaviorKernel<<<blocks, threadsPerBlock>>>(concentrations, expectations,
                                                variance, numSimulations,
                                                numTimes);
}
