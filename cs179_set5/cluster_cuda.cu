#include <cassert>
#include <cuda_runtime.h>
#include <float.h>
#include "cluster_cuda.cuh"

// This assumes address stores the average of n elements atomically updates
// address to store the average of n + 1 elements (the n elements as well as
// val). This might be useful for updating cluster centers.
// modified from http://stackoverflow.com/a/17401122
__device__ 
float atomicUpdateAverage(float* address, int n, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    float next_val = (n * __int_as_float(assumed) + val) / (n + 1);
    old = ::atomicCAS(address_as_i, assumed,
		      __float_as_int(next_val));
  } while (assumed != old);
  return __int_as_float(old);
}

// computes the distance squared between vectors a and b where vectors have
// length size and stride stride.
__device__ 
float squared_distance(float *a, float *b, int stride, int size) {
  float dist = 0.0;
  for (int i=0; i < size; i++) {
    float diff = a[stride * i] - b[stride * i];
    dist += diff * diff;
  }
  return dist;
}

/*
 * Notationally, all matrices are column majors, so if I say that matrix Z is
 * of size m * n, then the stride in the m axis is 1. For purposes of
 * optimization (particularly coalesced accesses), you can change the format of
 * any array.
 *
 * clusters is a REVIEW_DIM * k array containing the location of each of the k
 * cluster centers.
 *
 * cluster_counts is a k element array containing how many data points are in 
 * each cluster.
 *
 * k is the number of clusters.
 *
 * data is a REVIEW_DIM * batch_size array containing the batch of reviews to
 * cluster. Note that each review is contiguous (so elements 0 through 49 are
 * review 0, ...)
 *
 * output is a batch_size array that contains the index of the cluster to which
 * each review is the closest to.
 *
 * batch_size is the number of reviews this kernel must handle.
 */
__global__
void sloppyClusterKernel(float *clusters, int *cluster_counts, int k, 
                          float *data, int *output, int batch_size) {
    // TODO: write me
    // Access one element of the batch at a time...
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < batch_size) {
        float *this_review = data[index * REVIEW_DIM];
        
        // Find the closest cluster
        int closest_cluster = 0;
        float smallest_distance = FLT_MAX;
        for (int i = 0; i < k; i++) {
            float *cluster = clusters[i * REVIEW_DIM];
            float distance = squared_distance(this_review, cluster, 1, REVIEW_DIM);
            if (distance < smallest_distance) {
                closest_cluster = i;
                smallest_distance = distance;
            }
        }
        
        // Assign this_review to the closest cluster
        output[index] = closest_cluster;
        
        // update said cluster
        float *cluster = clusters[closest_cluster * REVIEW_DIM];
        int cluster_size = cluster_counts[closest_cluster];
        for (int i = 0; i < REVIEW_DIM; i++) {
            float newAvg = atomicUpdateAverage(cluster[i], cluster_size, this_review[i]);
            cluster[i] = newAvg;
        }
        
        // Update the cluster size
        cluster_counts[closest_cluster] = cluster_size + 1;        
        
        index += blockDim.x * gridDim.x;
    }
}


void cudaCluster(float *clusters, int *cluster_counts, int k,
		 float *data, int *output, int batch_size, 
		 cudaStream_t stream) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = 0;

  sloppyClusterKernel<<<
    block_size, 
    grid_size, 
    shmem_bytes, 
    stream>>>(clusters, cluster_counts, k, data, output, batch_size);
}
