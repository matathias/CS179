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

  __shared__ float data[4160];
  __syncthreads();

  // Calculate the indices
  int i = (threadIdx.x * 4) % 64;
  int global_i = i + 64 * blockIdx.x;
  int j = (4 * threadIdx.y) + (threadIdx.x / 16);
  int global_j = j + 64 * blockIdx.y;

  // Coalesced load values into shared memory from global memory, using a stride
  // length of 1 for both memory types.
  for (int iter = 0; iter < 4; iter++) {
    data[(i + iter) + 65 * j] = input[(global_i + iter) + n * global_j];
  }

  __syncthreads();

  // Re-calculate the indices for global memory
  global_i = i + 64 * blockIdx.y;
  global_j = j + 64 * blockIdx.x;
  // Coalesced write values into global memory from shared memory, using a
  // stride length of 1 for global memory and 65 for shared memory. This use of
  // padding should remove bank conflicts.
  for (int iter = 0; iter < 4; iter++) {
    output[(global_i + iter) + n * global_j] = data[j + 65 * (i + iter)];
  }
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {

  __shared__ float data[4160];
  // removed unnecessary __syncthreads() call

  // Calculate the indices
  int i = (threadIdx.x * 4) % 64;
  int global_i = i + 64 * blockIdx.x;
  int j = (4 * threadIdx.y) + (threadIdx.x / 16);
  int global_j = j + 64 * blockIdx.y;

  // Coalesced load values into shared memory from global memory, using a stride
  // length of 1 for both memory types.
  // "Unrolled" loop to reduce bounds checking overhead
  data[i + 65 * j] = input[global_i + n * global_j];
  data[i + 1 + 65 * j] = input[global_i + 1 + n * global_j];
  data[i + 2 + 65 * j] = input[global_i + 2 + n * global_j];
  data[i + 3 + 65 * j] = input[global_i + 3 + n * global_j];

  __syncthreads();

  // Re-calculate the indices for global memory
  global_i = i + 64 * blockIdx.y;
  global_j = j + 64 * blockIdx.x;
  // Coalesced write values into global memory from shared memory, using a
  // stride length of 1 for global memory and 65 for shared memory. This use of
  // padding should remove bank conflicts.
  // "Unrolled" loop to reduce bounds checking overhead
  output[global_i + n * global_j] = data[j + 65 * i];
  output[global_i + 1 + n * global_j] = data[j + 65 * i + 65];
  output[global_i + 2 + n * global_j] = data[j + 65 * i + 130];
  output[global_i + 3 + n * global_j] = data[j + 65 * i + 195];
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
