#include <cassert>
#include <cuda_runtime.h>
#include "transpose_cuda.cuh"

/**
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
*/


/**
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304  matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
  // TODO: do not modify code, just comment on suboptimal accesses

  // As each warp handles a 32x4 submatrix, it is impossible for this method to
  // have coalesced writes as the data handled by each warp is spread across
  // different cache lines.
  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  for (; j < end_j; j++) {
    output[j + n * i] = input[i + n * j];
  }
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
  // TODO: Modify transpose kernel to use shared memory. All global memory
  // reads and writes should be coalesced. Minimize the number of shared
  // memory bank conflicts (0 bank conflicts should be possible using
  // padding). Again, comment on all sub-optimal accesses.

  __shared__ float data[4096];
  __syncthreads();
/*
  // Load data from global memory into shared memory
  const int i = 4 * threadIdx.x;
  int j = 64 * threadIdx.y;
  int shmemLoadStart = i + j;
  int globalLoadStart = i + j + (64 * blockIdx.x) + (64 * blockIdx.y);

  // Load from global memory with stride 1 into shared memory with stride 1
  for (int iter = 0; iter < 4; iter++) {
    data[shmemLoadStart + iter] = input[globalLoadStart + iter];
  }

  __syncthreads();

  // Load data from shared memory into global memory in such a way that
  // transposes the matrix
  int shmemInd = 64 * threadIdx.x + 4 * threadIdx.y;
  for (int iter = 0; iter < 4; iter++) {
    if (shmemInd % 4096 < shmemInd) {
      shmemInd = (shmemInd % 4096) + 1;
    }
    output[globalLoadStart + iter] = data[shmemInd];
    shmemInd += n;
  } */
  int i = (threadIdx.x * 4) % 64;
  const int global_i = i + 64 * blockIdx.x;
  int j = (4 * threadIdx.y) + (threadIdx.x / 16);
  const int global_j = j + 64 * blockIdx.y;

  for (int iter = 0; iter < 4; iter++) {
    data[(i + iter) + 64 * j] = input[(global_i + iter) + n * global_j];
  }

  __syncthreads();

  
  for (int iter = 0; iter < 4; iter++) {
    output[(global_i + iter) + n * global_j] = data[j + 64 * (i + iter)];
  }
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
  // TODO: This should be based off of your shmemTransposeKernel.
  // Use any optimization tricks discussed so far to improve performance.
  // Consider ILP and loop unrolling.

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  for (; j < end_j; j++) {
    output[j + n * i] = input[i + n * j];
  }
}

void cudaTranspose(const float *d_input,
                   float *d_output,
                   int n,
                   TransposeImplementation type) {
  if (type == NAIVE) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == SHMEM) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == OPTIMAL) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else {
    // unknown type
    assert(false);
  }
}
