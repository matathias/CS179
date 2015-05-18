#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
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

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
  }

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
  // seed generator to 2015
  std::default_random_engine generator(2015);
  std::normal_distribution<float> distribution(0.0, 0.1);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[stride * component_idx] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {
  // TODO: randomly initialize weights, allocate and initialize buffers on
  //       host & device
  printf("Entered classify()\n");
  
  cudaEvent_t stopEvent, startEvent;
  gpuErrChk(cudaEventCreate(&startEvent));
  gpuErrChk(cudaEventCreate(&stopEvent));
  
  float *weights = (float*)malloc(sizeof(float) * REVIEW_DIM);
  gaussianFill(weights, REVIEW_DIM);
  
  // Allocate memory on host for LSAReviews
  float *data = (float*)malloc(sizeof(float) * batch_size * (REVIEW_DIM + 1));
  
  // Allocate memory on device for LSAReviews
  float *d_data;
  gpuErrChk(cudaMalloc(&d_data, batch_size * (REVIEW_DIM + 1) *
                                sizeof(float)));
                                
  // Allocate memory on device for weights
  float *d_weights;
  gpuErrChk(cudaMalloc(&d_weights, sizeof(float) * REVIEW_DIM));
  
  int step_size = 0.1;
  float classification_time = -1;

  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0;
  int batch_number = 0;
  
  // Record how long the entire process takes, to compare how long IO takes to
  // how long the kernal takes
  START_TIMER();
  
  for (string review_str; getline(in_stream, review_str); review_idx++) {
    // TODO: process review_str with readLSAReview
    int data_idx = review_idx % batch_size;
    readLSAReview(review_str, &data[data_idx], 1);

    // TODO: if batch is full, call kernel
    // -1 is to account for the fact that review_idx is 0-indexed.
    if (review_idx % batch_size == batch_size - 1) {
        // Copy H->D, call kernal, copy D->H
        gpuErrChk(cudaEventRecord(startEvent, 0));
        gpuErrChk(cudaMemcpy(d_data, data, 
                             batch_size * (REVIEW_DIM + 1) * sizeof(float), 
                             cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(d_weights, weights, REVIEW_DIM * sizeof(float),
                             cudaMemcpyHostToDevice));
                                  
        float errors = cudaClassify(d_data, batch_size, step_size, d_weights);
        errors = errors * 100; //turn it into a percentage
                    
        gpuErrChk(cudaMemcpy(weights, d_weights, REVIEW_DIM * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        gpuErrChk(cudaEventRecord(stopEvent, 0));
        
        // Print the batch number and the error rate
        printf("\nBatch Number: %d\n", batch_number);
        printf("Batch Error Rate: %f percent\n", errors);
        
        batch_number++;
    }
    
  }
  
  // We aren't doing any streams shenanigans currently, but just in case...
  gpuErrChk(cudaDeviceSynchronize());
  
  STOP_RECORD_TIMER(classification_time);
  
  // Transform classification_time into seconds from milliseconds
  classification_time = classification_time / 1000;
  
  // Find the elapsed time for the kernal
  float time;
  gpuErrChk(cudaEventElapsedTime(&time, startEvent, stopEvent));
  
  // Transform time into seconds from milliseconds
  time = time / 1000;
  
  // Calculate the kernal throughput
  float throughput = batch_size / time;

  // TODO: print out weights
  printf("\nWeights:\n[");
  for (int i = 0; i < REVIEW_DIM; i++) {
    printf("%.4e,", weights[i]);
  }
  printf("]\n\n");
  
  // Print out the timing and throughput data
  printf("Time to classify all reviews:      %f seconds\n",classification_time);
  printf("Batch size:                        %d\n", batch_size);
  printf("Number of batches:                 %d\n", batch_number);
  printf("Single kernal latency:             %f\n", time);
  printf("Kernal Throughput (reviews/s):     %f\n", throughput);
  printf("Total kernal latency (calculated): %f\n", time * batch_number);
  printf("IO time (calculated):              %f\n", classification_time - 
                                                    (time * batch_number));
  
  // TODO: free all memory
  free(weights);
  free(data);
  gpuErrChk(cudaFree(d_weights));
  gpuErrChk(cudaFree(d_data));
  gpuErrChk(cudaEventDestroy(startEvent));
  gpuErrChk(cudaEventDestroy(stopEvent));
}

int main(int argc, char** argv) {
  int batch_size = 2048;
  
  if (argc == 1) {
    classify(cin, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);
  }
}
