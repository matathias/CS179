#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "cluster_cuda.cuh"

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
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM floats.
void readLSAReview(string review_str, float *output) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[component_idx] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM);
}

// used to pass arguments to printerCallback
struct printerArg {
  int review_idx_start;
  int batch_size;
  int *cluster_assignments;
};

// Prints out which cluster each review in a batch was assigned to.
// TODO: Call with cudaStreamAddCallback (after completing D->H memcpy)
void printerCallback(cudaStream_t stream, cudaError_t status, void *userData) {
  printerArg *arg = static_cast<printerArg *>(userData);

  for (int i=0; i < arg->batch_size; i++) {
    printf("%d: %d\n", 
	   arg->review_idx_start + i, 
	   arg->cluster_assignments[i]);
  }

  delete arg;
}

void cluster(istream& in_stream, int k, int batch_size) {
  // cluster centers
  float *d_clusters;

  // how many points lie in each cluster
  int *d_cluster_counts;

  // allocate memory for cluster centers and counts
  gpuErrChk(cudaMalloc(&d_clusters, k * REVIEW_DIM * sizeof(float)));
  gpuErrChk(cudaMalloc(&d_cluster_counts, k * sizeof(int)));

  // randomly initialize cluster centers
  float *clusters = new float[k * REVIEW_DIM];
  gaussianFill(clusters, k * REVIEW_DIM);
  gpuErrChk(cudaMemcpy(d_clusters, clusters, k * REVIEW_DIM * sizeof(float),
		       cudaMemcpyHostToDevice));

  // initialize cluster counts to 0
  gpuErrChk(cudaMemset(d_cluster_counts, 0, k * sizeof(int)));
  
  float classification_time = -1;
  
  // TODO: allocate copy buffers and streams
  int numStreams = 2;
  cudaStream_t s[numStreams];
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&s[i]);
  }
  // Allocate memory on host for LSAReviews
  float *data = new float[numStreams * batch_size * REVIEW_DIM];
  
  // Allocate memory on device for LSAReviews
  float *d_data;
  gpuErrChk(cudaMalloc(&d_data, 
                       numStreams * batch_size * REVIEW_DIM * sizeof(float)));
  
  // Allocate memory on host for the output
  int *output = new int[numStreams * batch_size];
  
  // Allocate memory on device for the output
  int *d_output;
  gpuErrChk(cudaMalloc(&d_output, numStreams * batch_size * sizeof(int)));

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  gpuErrChk(cudaEventCreate(&startEvent));
  gpuErrChk(cudaEventCreate(&stopEvent));
  float time;
  float singleBatchTime;

  // Record how long it takes to classify everything
  START_TIMER();
  
  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0;
  for (string review_str; getline(in_stream, review_str); review_idx++) {
    // TODO: readLSAReview into appropriate storage
    int data_idx = review_idx % (batch_size * numStreams);
    readLSAReview(review_str, &data[data_idx]);

    // TODO: if you have filled up a batch, copy H->D, kernel, copy D->H,
    //       and set callback to printerCallback. Will need to allocate
    //       printerArg struct. Do all of this in a stream.
    
    // -1 is to account for the fact that review_idx is 0-indexed.
    if (review_idx % batch_size == batch_size - 1) {
        int stream_idx = (review_idx / batch_size) % numStreams;
        int offset = stream_idx * batch_size;
        
        // Copy H->D, call kernal, copy D->H
        gpuErrChk(cudaEventRecord(startEvent, 0));
        gpuErrChk(cudaMemcpyAsync(&d_data[offset], &data[offset], 
                                  batch_size * REVIEW_DIM * sizeof(float), 
                                  cudaMemcpyHostToDevice, s[stream_idx]));
                                  
        cudaCluster(d_clusters, d_cluster_counts, k, 
                    &d_data[offset], &d_output[offset], batch_size, 
                    s[stream_idx]);
                    
        gpuErrChk(cudaMemcpyAsync(&output[offset], &d_output[offset], 
                                  batch_size * sizeof(float), 
                                  cudaMemcpyDeviceToHost, s[stream_idx]));
        gpuErrChk(cudaEventRecord(stopEvent, 0));
                        
        //initialize the printerArg struct
        struct printerArg *stream_printer = 
                                      (printerArg *) malloc(sizeof(printerArg));
        // review_idx is 0-indexed, so we have to account for that
        stream_printer->review_idx_start = (review_idx + 1) - batch_size;
        stream_printer->batch_size = batch_size;
        stream_printer->cluster_assignments = &output[offset];
        
        // Set a callback to printerCallback
        cudaStreamAddCallback(s[stream_idx], 
                              printerCallback, 
                              stream_printer, 0);
    }
  }

  // wait for everything to end on GPU before final summary
  gpuErrChk(cudaDeviceSynchronize());
  
  // Find the elapsed time for the batch
  gpuErrChk(cudaEventSynchronize(stopEvent));
  gpuErrChk(cudaEventElapsedTime(&time, startEvent, stopEvent));
  singleBatchTime = time / 1000;
  
  STOP_RECORD_TIMER(classification_time);

  float cCountBand = 0;
  float clusterBand = 0;

  // retrieve final cluster locations and counts
  int *cluster_counts = new int[k];
  gpuErrChk(cudaEventRecord(startEvent, 0));
  gpuErrChk(cudaMemcpy(cluster_counts, d_cluster_counts, k * sizeof(int), 
		       cudaMemcpyDeviceToHost));
  gpuErrChk(cudaEventRecord(stopEvent, 0));
  gpuErrChk(cudaEventSynchronize(stopEvent));
  gpuErrChk(cudaEventElapsedTime(&time, startEvent, stopEvent));
  cCountBand = k * sizeof(int) * 1e-6 / time;
  
  gpuErrChk(cudaEventRecord(startEvent, 0));
  gpuErrChk(cudaMemcpy(clusters, d_clusters, k * REVIEW_DIM * sizeof(int),
		       cudaMemcpyDeviceToHost));
  gpuErrChk(cudaEventRecord(stopEvent, 0));
  gpuErrChk(cudaEventSynchronize(stopEvent));
  gpuErrChk(cudaEventElapsedTime(&time, startEvent, stopEvent));
  clusterBand = k * REVIEW_DIM * sizeof(int) * 1e-6 / time;

  // print cluster summaries
  for (int i=0; i < k; i++) {
    printf("Cluster %d, population %d\n", i, cluster_counts[i]);
    printf("[");
    for (int j=0; j < REVIEW_DIM; j++) {
      printf("%.4e,", clusters[i * REVIEW_DIM + j]);
    }
    printf("]\n\n");
  }
  
  // Print how long classification took
  printf("Total classification time: %f seconds\n", classification_time / 1000);
  printf("Number of reviews:         %d\n", review_idx);
  printf("Reviews per second:        %f\n\n", 
         review_idx * 1000 / classification_time );
         
  printf("Batch size:           %d\n", batch_size);
  printf("k:                    %d\n", k);
  printf("Single batch latency: %f seconds\n\n", singleBatchTime);
  
  // Print the bandwidth used in the final copy
  printf("Device to Host cluster_counts bandwidth (GB/s): %f\n", cCountBand);
  printf("Device to Host clusters bandwidth (GB/s):       %f\n", clusterBand);

  // free cluster data
  gpuErrChk(cudaFree(d_clusters));
  gpuErrChk(cudaFree(d_cluster_counts));
  delete[] cluster_counts;
  delete[] clusters;

  // TODO: finish freeing memory, destroy streams
  gpuErrChk(cudaFree(d_data));
  gpuErrChk(cudaFree(d_output));
  gpuErrChk(cudaEventDestroy(startEvent));
  gpuErrChk(cudaEventDestroy(stopEvent));
  delete[] data;
  delete[] output;
  for (int i = 0; i < numStreams; i++) {
    gpuErrChk(cudaStreamDestroy(s[i]));
  }
}

int main(int argc, char** argv) {
  int k = 50;
  int batch_size = 4096;

  if (argc == 1) {
    cluster(cin, k, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    cluster(buffer, k, batch_size);
  }
}
