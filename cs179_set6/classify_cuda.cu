#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(float *data, int batch_size, int step_size,
		       float *weights, float *errors) {
  // TODO: write me
  extern __shared__ float shData[];
  // Copy weights into shared memory (the first row of data[])
  unsigned int index = threadIdx.x;
  while (index < REVIEW_DIM) {
    shData[index] = weights[index];
    shData[index + REVIEW_DIM] = 0;
    index += blockDim.x;
  }
  
  __syncthreads();
  
  // Access one element of the batch at a time
  unsigned int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  while (batch_index < batch_size) {
    float *this_review = &data[(REVIEW_DIM + 1) * batch_index];
    float y_n = this_review[REVIEW_DIM];
    float wTxN = 0;
    
    // First calculate w^T * x_n
    for (int i = 0; i < REVIEW_DIM; i++) {
        wTxN += shData[i] * this_review[i];
    }
    
    // Use this value as a prediction to calculate the error value
    // if wTxN * y_n is positive, then they have the same sign. Otherwise they
    // don't, and a misclassification error has occured.
    if (wTxN * y_n < 0) {
        atomicAdd(errors, 1);
    }
    
    // Now calculate 1 + exp(y_n * w^T * x_n)
    float factor = 1.0 + exp(y_n * wTxN);
    
    // Now calculate the (y_n * x_n) part. Divide each individual value by
    // divisor and store in the appropriate location in shData, which will be
    // the second row (the first row is occupied by the weights)
    for (int i = 0; i < REVIEW_DIM; i++) {
        float val = y_n * this_review[i];
        val = (val / factor) * (-1.0 / ((float) batch_size));
        shData[REVIEW_DIM + i] += val;
    }
    
    batch_index += blockDim.x * gridDim.x;
  }
  
  __syncthreads();
  // Add the thread's gradient contribution to the weights and return
  index = threadIdx.x;
  while (index < REVIEW_DIM) {
    // The gradient contribution is on the second row of data
    atomicAdd(&weights[index], -1 * step_size * shData[index + REVIEW_DIM]);
    index += blockDim.x;
  }
  
  // We only want to divide errors by the batch size once...
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 1) {
    *errors = *errors / batch_size;
  }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(float *data, int batch_size,
                   float step_size, float *weights) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = REVIEW_DIM * 2 * sizeof(float);

  float *d_errors;
  cudaMalloc(&d_errors, sizeof(float));
  cudaMemset(d_errors, 0, sizeof(float));

  trainLogRegKernel<<<grid_size, block_size, shmem_bytes>>>(data,
                                                            batch_size,
                                                            step_size,
                                                            weights,
                                                            d_errors);
  float h_errors = -1.0;
  cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
  cudaFree(d_errors);
  return h_errors;
}
