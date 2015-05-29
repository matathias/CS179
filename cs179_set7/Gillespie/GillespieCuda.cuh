

void callGillespieKernel(int *productionStates, 
                         int *old_concentrations, int *new_concentrations,
                         float *times, float *randomTimeSteps,
                         float *randomProbs, curandState_t *state,
                         int numSimulations,
                         int blocks, int threadsPerBlock);

void callResampleKernel(int *concentrations, int *new_concentrations,
                        float *times, int numSimulations, float timeFactor,
                        float endTime, int *done, int blocks, 
                        int threadsPerBlock);

void callBehaviorKernel(int *concentrations, float *expectations, 
                        float *variance, int numSimulations, int numTimes,
                        int blocks, int threadsPerBlock);
